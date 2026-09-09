"""Theoretical quantiles of the simulated processes."""

import numpy as np
from scipy.stats import levy_stable

from src.stable_mar.stable_mar import madelta


def compute_theoretical_quantiles(
    phi_vec,
    psi_vec,
    alpha,
    beta,
    sigma,
    theta=None,
    eta=None,
    ma_trunc=100,
):
    """Compute lower/upper marginal quantiles of a stable MAR process.

    `phi_vec`/`psi_vec` are the causal/non-causal MA coefficients; `alpha`,
    `beta`, `sigma` are the stable-innovation parameters. Returns a dict with
    'Lower Quantiles' and 'Upper Quantiles', each mapping levels to values
    (zero location shift).
    """
    coeff_ma = np.array(
        [
            madelta(cvec=phi_vec, ncvec=psi_vec, k=k, theta=theta, eta=eta)
            for k in range(-ma_trunc, ma_trunc)
        ],
        dtype=float,
    )
    coeff_ma = np.real(coeff_ma)

    # Compute scale and skewness-adjusted parameters
    abs_ma_alpha = np.abs(coeff_ma) ** alpha
    sig1 = (sigma**alpha) * np.sum(abs_ma_alpha)
    bet1 = beta * np.sum(np.sign(coeff_ma) * abs_ma_alpha) / np.sum(abs_ma_alpha)
    scale = sig1 ** (1 / alpha)

    # Quantile probabilities
    lower_probs = [0.1, 0.05, 0.01, 0.005, 0.001]
    upper_probs = [0.9, 0.95, 0.99, 0.995, 0.999]

    # Compute quantiles
    lower_quantiles = levy_stable.ppf(lower_probs, alpha, bet1, loc=0, scale=scale)
    upper_quantiles = levy_stable.ppf(upper_probs, alpha, bet1, loc=0, scale=scale)

    return {
        "Lower Quantiles": dict(zip(lower_probs, lower_quantiles)),
        "Upper Quantiles": dict(zip(upper_probs, upper_quantiles)),
    }
