"""Markov-switching Gaussian bubble DGP with exact predictive densities.

Three-regime Markov-switching Gaussian AR(1) in the observed lag:

    y_t = mu(S_t) + phi(S_t) * y_{t-1} + sigma(S_t) * eps_t,   eps_t ~ iid N(0,1)

    regime      mu      phi     sigma      role
    B (0)       0       phi0<1  sigma_b    calm mean reversion around 0
    U (1)       +delta  rho>1   sigma_e    mildly explosive, upward
    D (2)       -delta  rho>1   sigma_e    mildly explosive, downward

    transitions: B->U, B->D with prob q each (onset every 1/(2q) periods on
    average); U->B, D->B with prob `hazard` (geometric bubble duration). When
    a bubble ends the chain returns directly to B, so the level mean-reverts
    back toward 0 at geometric rate phi0.
"""

import numpy as np

# Regime indices
B, U, D = 0, 1, 2

_LOG_2PI = np.log(2.0 * np.pi)


class MarkovSwitchingBubble:
    """Simulator + exact filter + closed-form h-step predictive density."""

    def __init__(
        self,
        phi0=0.7,
        sigma_b=1.0,
        q=0.0025,
        delta=4.0,
        rho=1.02,
        sigma_e=1.0,
        hazard=0.05,
        component_tol=1e-10,
    ):
        """Set the regime parameters; see the class docstring."""
        if not 0.0 < phi0 < 1.0:
            raise ValueError(f"phi0 must be in (0, 1), got {phi0}")
        if not 0.0 < 2.0 * q < 1.0:
            raise ValueError(f"q must be in (0, 0.5), got {q}")
        if rho <= 1.0:
            raise ValueError(f"rho must exceed 1 (explosive regime), got {rho}")
        if not 0.0 < hazard < 1.0:
            raise ValueError(f"hazard must be in (0, 1), got {hazard}")
        if min(sigma_b, sigma_e) <= 0.0 or delta <= 0.0:
            raise ValueError("sigma_b, sigma_e and delta must be positive")
        if phi0 * rho ** (2.0 * q / hazard) >= 1.0:
            raise ValueError(
                "non-stationary parametrization: need phi0 * rho^(2q/hazard) < 1, "
                f"got {phi0 * rho ** (2.0 * q / hazard):.4f}"
            )

        self.q = float(q)
        self.hazard = float(hazard)
        self.component_tol = float(component_tol)

        # Per-regime emission coefficients, indexed by (B, U, D).
        self.mu = np.array([0.0, delta, -delta])
        self.phi = np.array([phi0, rho, rho])
        self.sigma = np.array([sigma_b, sigma_e, sigma_e])

        self.P = np.array(
            [
                [1.0 - 2.0 * q, q, q],
                [hazard, 1.0 - hazard, 0.0],
                [hazard, 0.0, 1.0 - hazard],
            ]
        )

        # Marginal tail index alpha
        self.tail_index = np.log(1.0 / (1.0 - hazard)) / np.log(rho)

        self._components_cache = {}
        self._marginal_cache = {}

    # Simulation
    def simulate(self, n, seed=None):
        """Draw one trajectory of length `n`.

        Index = time: y[0] is the known initial condition y_0 = 0 with S_0 = B
        (the convention the exact filter/predictive density relies on); y[1:]
        are the stochastic observations.
        """
        rng = np.random.default_rng(seed)
        u = rng.random(n)
        eps = rng.standard_normal(n)

        y = np.empty(n)
        y[0] = 0.0

        mu, phi, sig = self.mu, self.phi, self.sigma
        q, hazard = self.q, self.hazard
        r = B
        for t in range(1, n):
            if r == B:
                if u[t] < q:
                    r = U
                elif u[t] < 2.0 * q:
                    r = D
            elif u[t] < hazard:
                r = B
            y[t] = mu[r] + phi[r] * y[t - 1] + sig[r] * eps[t]

        return y

    # Exact Hamilton filter
    def filter_probabilities(self, y):
        """Exact filtered regime probabilities xi[t, k] = P(S_t = k | y_{0:t}).

        `y` follows the simulate() convention (y[0] = 0, S_0 = B known), so the
        recursion starts from a point mass on B. The update is done on
        max-shifted log weights, so bubble-scale residuals cannot underflow the
        normalization.
        """
        y = np.asarray(y, dtype=np.float64)
        n = len(y)
        log_norm = -0.5 * _LOG_2PI - np.log(self.sigma)
        inv_sig = 1.0 / self.sigma

        xi = np.zeros((n, 3))
        xi[0, B] = 1.0
        for t in range(1, n):
            prior = xi[t - 1] @ self.P
            z = (y[t] - self.mu - self.phi * y[t - 1]) * inv_sig
            loglik = log_norm - 0.5 * z * z
            with np.errstate(divide="ignore"):
                logw = np.log(prior) + loglik
            w = np.exp(logw - logw.max())
            w[w < 1e-280] = 0.0  # flush denormals (max entry is 1 by construction)
            xi[t] = w / w.sum()
        return xi

    # Closed-form h-step predictive density
    def predictive_components(self, horizon):
        """Exact mixture components of f(y_{t+h} | y_{0:t}) for h = `horizon`.

        Conditional on a regime path s_{t+1:t+h}, y_{t+h} | y_t is
        N(a + b * y_t, v) with path-specific (a, b, v); the path weight is
        xi_pred(k1) * c, where k1 is the path's first regime, c the product of
        its transition probabilities, and xi_pred = xi_{t|t} @ P. Components
        with identical (k1, current regime, a, b, v) are merged by summing c
        (an exact mixture identity). The count grows like (1 + sqrt(2))^h --
        3 at h = 1, ~8,100 at h = 10.

        Paths whose probability product c falls below `component_tol` are
        discarded and the surviving weights renormalized per first regime:
        they are paths with 3+ regime switches inside the horizon, and the
        total discarded probability is bounded by (count * tol) ~ 1e-6 at the
        default tol -- far below any evaluation metric's resolution. Set
        component_tol = 0 at construction for the fully exact enumeration.

        Returns (k1, a, b, v, c) as arrays over components, cached per horizon.
        """
        if horizon in self._components_cache:
            return self._components_cache[horizon]
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")

        mu, phi, sig2 = self.mu, self.phi, self.sigma**2
        tol = self.component_tol
        comps = {(k, k, mu[k], phi[k], sig2[k]): 1.0 for k in range(3)}
        for _ in range(horizon - 1):
            new = {}
            for (k1, r, a, b, v), c in comps.items():
                for k in range(3):
                    p = self.P[r, k]
                    if p == 0.0:
                        continue
                    cp = c * p
                    if cp < tol:
                        continue
                    key = (
                        k1,
                        k,
                        mu[k] + phi[k] * a,
                        phi[k] * b,
                        phi[k] ** 2 * v + sig2[k],
                    )
                    new[key] = new.get(key, 0.0) + cp
            comps = new

        k1 = np.array([key[0] for key in comps], dtype=np.int64)
        a = np.array([key[2] for key in comps])
        b = np.array([key[3] for key in comps])
        v = np.array([key[4] for key in comps])
        c = np.array(list(comps.values()))
        for k in range(3):
            c[k1 == k] /= c[k1 == k].sum()
        self._components_cache[horizon] = (k1, a, b, v, c)
        return self._components_cache[horizon]

    def _mixture(self, y, cond_idx, horizon, xi=None):
        """Per-observation mixture (w, m, v) of f(y_{tau+h} | y_{0:tau}).

        One row per conditioning time tau in ``cond_idx``.
        """
        y = np.asarray(y, dtype=np.float64)
        cond_idx = np.asarray(cond_idx, dtype=np.int64)
        if xi is None:
            xi = self.filter_probabilities(y)

        k1, a, b, v, c = self.predictive_components(horizon)
        xi_pred = (xi[cond_idx, :, None] * self.P[None, :, :]).sum(axis=1)
        w = xi_pred[:, k1] * c  # (n, n_components)
        np.testing.assert_allclose(w.sum(axis=1), 1.0, rtol=1e-8)
        m = a[None, :] + b[None, :] * y[cond_idx, None]
        return w, m, v

    def predictive_density(self, y, cond_idx, horizon, grid, xi=None):
        """Exact predictive density f(y_{tau+h} | y_{0:tau}) on `grid`.

        `y` is the full observed path (simulate() convention), `cond_idx` the
        conditioning time indices tau (the density conditions on y[0..tau] and
        targets y[tau + horizon]). Pass a precomputed `xi` from
        filter_probabilities() to skip re-filtering. Returns
        (len(cond_idx), len(grid)).
        """
        grid = np.asarray(grid, dtype=np.float64)
        w, m, v = self._mixture(y, cond_idx, horizon, xi=xi)
        return self._sum_gaussian_mixture(w, m, v, grid)

    @staticmethod
    def _sum_gaussian_mixture(w, m, v, grid):
        """Row-wise mixture: out[i, g] = sum_j w[i,j] N(grid[g]; m[i,j], v[j]).

        The component count grows quickly with the horizon (~8,100 raw paths at
        h = 10), so the exp-heavy sum runs through torch when it is installed
        (multithreaded on CPU, CUDA when available; float64 throughout, so the
        result is identical to the numpy path); otherwise a plain numpy loop.
        """
        inv_sd = 1.0 / np.sqrt(v)
        coef = w * (inv_sd / np.sqrt(2.0 * np.pi))[None, :]  # (n, K)
        try:
            import torch
        except ImportError:
            out = np.zeros((m.shape[0], len(grid)))
            for j in range(w.shape[1]):
                if not w[:, j].any():
                    continue
                z = (grid[None, :] - m[:, j][:, None]) * inv_sd[j]
                out += coef[:, j][:, None] * np.exp(-0.5 * z * z)
            return out

        device = "cuda" if torch.cuda.is_available() else "cpu"
        g = torch.as_tensor(grid, dtype=torch.float64, device=device)
        m_t = torch.as_tensor(m * inv_sd[None, :], dtype=torch.float64, device=device)
        s_t = torch.as_tensor(inv_sd, dtype=torch.float64, device=device)
        c_t = torch.as_tensor(coef, dtype=torch.float64, device=device)
        out = torch.zeros((m.shape[0], len(g)), dtype=torch.float64, device=device)
        chunk = max(1, int(4e7 // (m.shape[0] * len(grid))) or 1)
        for j0 in range(0, m.shape[1], chunk):
            j1 = min(j0 + chunk, m.shape[1])
            z = g[None, None, :] * s_t[None, j0:j1, None] - m_t[:, j0:j1, None]
            out += (c_t[:, j0:j1, None] * torch.exp(z.mul_(z).mul_(-0.5))).sum(dim=1)
        return out.cpu().numpy()

    # Marginal quantiles (no closed form: long-simulation estimate)
    def marginal_quantiles(self, levels, n=2_000_000, seed=0, burn=10_000):
        """Stationary marginal quantiles at ``levels``, from one long simulation.

        Cached. The marginal law has a Pareto tail of index ``self.tail_index``,
        so keep the levels well inside (0, 1).
        """
        key = (n, seed, burn)
        if key not in self._marginal_cache:
            self._marginal_cache[key] = self.simulate(n, seed=seed)[burn:]
        return np.quantile(self._marginal_cache[key], np.asarray(levels))
