"""Alpha-stable MAR simulation and estimation."""

import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats


class stablemar:
    """
    Class representing Mixed Autoregressive (MAR) models with alpha-stable innovations.

    The MAR(r,s) model is defined as:
    Ψ(F)Φ(B)X_t = ε_t

    where:
    - Ψ(F) is the noncausal polynomial with F the forward operator
    - Φ(B) is the causal polynomial with B the backward operator
    - ε_t follows an alpha-stable distribution S(α, β, σ, 0)

    Attributes:
        order (Tuple[int, int]): Orders (r, s) of the causal and noncausal lags
        par (List[float]): List of model parameters
        results (Dict[str, Any]): Dictionary to store estimation results
        trajectory (pd.Series): Simulated or filtered trajectory
        innovation (pd.Series): Innovations for simulated trajectory
    """

    def __init__(self, order: Tuple[int, int], par: List[float] = None):
        """Set the model order (r, s) and, optionally, its parameters."""
        self.order = order
        self.par = par if par is not None else []
        self.results = {}
        self.trajectory = None
        self.innovation = None

    def generate(
        self, n: int, errors: List[float] = None, seed: Optional[int] = None
    ) -> "stablemar":
        """Simulate the MAR(r, s), setting the trajectory and innovations."""
        if self.par is None or len(self.par) < 3:
            raise ValueError("Parameters must be set before generating data")

        r, s = self.order

        # Extract stable distribution parameters
        if len(self.par) > r + s:
            alpha = self.par[r + s]
            beta = self.par[r + s + 1] if len(self.par) > r + s + 1 else 0.0
            sigma = self.par[r + s + 2] if len(self.par) > r + s + 2 else 1.0
        else:
            alpha, beta, sigma = 1.5, 0.0, 1.0  # Default values

        # Set random seed
        if seed is not None:
            np.random.seed(seed)

        # Generate innovations
        m = 50  # Truncation for the MA filter
        if n < 2 * m:
            warnings.warn(f"Sample size (n={n}) is too small... n >= 100 is required")

        ntilde = n + 2 * m + 1

        # Get the MA filter coefficients
        deltas = self.ma_filter(m)
        deltas = np.flip(deltas)  # Respecter l'orientation de l'original

        # Generate alpha-stable errors
        if errors is None or len(errors) != ntilde:
            esim = stats.levy_stable.rvs(
                alpha=alpha, beta=beta, scale=sigma, loc=0, size=n * 3
            )
        else:
            esim = errors

        # Generate the process using MA filter
        xsim = np.ones(ntilde) * esim[0]

        for t in range(m, ntilde):
            xsim[t] = np.sum(deltas * esim[t - m : t + m])

        xsim = xsim[m:-m]
        esim = esim[m:-m]

        # Store results
        self.trajectory = pd.Series(xsim)
        self.innovation = pd.Series(esim)

        return self

    def ma_filter(self, m: int) -> np.ndarray:
        """MA filter coefficients, truncated at ``m``."""
        if self.par is None:
            raise ValueError("Parameters must be set before generating MA filter")

        r, s = self.order

        # Extract MAR parameters
        if r > 0:
            psi = np.array(self.par[:r])
        else:
            psi = np.array([])

        if s > 0:
            phi = np.array(self.par[r : r + s])
        else:
            phi = np.array([])

        deltas = np.full(2 * m, np.nan)

        for k in range(-m, m):
            deltas[k + m] = madelta(psi, phi, k)

        deltas = np.flip(deltas)

        return deltas


# Helper functions
def madelta(
    cvec: np.ndarray, ncvec: np.ndarray, k: int, theta: float = None, eta: float = None
) -> float:
    """Infinite-MA coefficient at lag ``k``.

    ``theta``/``eta`` None or 0 means a pure MAR.
    """
    # Check if MARMA or MAR
    is_marma = (theta is not None and theta != 0) or (eta is not None and eta != 0)

    if not is_marma:
        # MAR process
        if len(cvec) > 0 and not np.all(cvec == 0):
            lam = 1 / np.roots(np.flip(np.concatenate(([1], -np.array(cvec)))))
            lam = lam.real
            r = len(lam)
        else:
            lam = 0
            r = 0
        if len(ncvec) > 0 and not np.all(ncvec == 0):
            zeta = 1 / np.roots(np.flip(np.concatenate(([1], -np.array(ncvec)))))
            zeta = zeta.real
            s = len(zeta)
        else:
            zeta = 0
            s = 0
        delta = 0
        if k >= 0:
            for j in range(s):
                numerator = zeta[j] ** ((s - 1) + k)
                denominator1 = (
                    1
                    if s == 1
                    else np.prod([zeta[j] - zeta[i] for i in range(s) if i != j])
                )
                denominator2 = np.prod([zeta[j] * lam[i] - 1 for i in range(r)])
                if s % 2 == 0:
                    delta -= numerator / (denominator1 * denominator2)
                else:
                    delta += numerator / (denominator1 * denominator2)
        else:
            for j in range(r):
                numerator = lam[j] ** ((r - 1) - k)
                denominator1 = (
                    1
                    if r == 1
                    else np.prod([lam[j] - lam[i] for i in range(r) if i != j])
                )
                denominator2 = np.prod([lam[j] * zeta[i] - 1 for i in range(s)])
                if r % 2 == 0:
                    delta -= numerator / (denominator1 * denominator2)
                else:
                    delta += numerator / (denominator1 * denominator2)
        if (r % 2 != 0) and (s % 2 != 0):
            delta = -delta
        elif (r % 2 == 0) and (s % 2 == 0):
            delta = -delta
        return delta

    else:
        # MARMA process: Fries code
        # First compute MAR coefficient at lag k
        mar_coeff_k = madelta(cvec, ncvec, k, theta=None, eta=None)

        # MARMA[k] = (1+theta*eta)*MAR[k] - theta*MAR[k-1] - eta*MAR[k+1]
        mar_coeff_k_minus_1 = madelta(cvec, ncvec, k - 1, theta=None, eta=None)
        mar_coeff_k_plus_1 = madelta(cvec, ncvec, k + 1, theta=None, eta=None)

        marma_coeff = (
            (1 + theta * eta) * mar_coeff_k
            - theta * mar_coeff_k_minus_1
            - eta * mar_coeff_k_plus_1
        )

        return marma_coeff
