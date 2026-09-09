"""Density-vs-density metrics (ISE, squared Hellinger, forward KL).

Also the mode/mean/median point estimates. Densities are renormalized over the
finite grid first, so scores reflect shape error rather than tail truncation.
"""

import numpy as np

from src.metrics.common import normalize


def region_means(integrand, grid_y, inside_sample_mask):
    """Integrate over the whole grid, then average across samples per region."""
    per_sample = np.trapezoid(integrand, grid_y, axis=1)

    def _m(a):
        return float(a.mean()) if a.size else float("nan")

    return {
        "all": _m(per_sample),
        "inside": _m(per_sample[inside_sample_mask]),
        "outside": _m(per_sample[~inside_sample_mask]),
    }


def ise_regions(pred_density, true_dens, grid_y, inside_sample_mask):
    """Return the mean integrated squared error per region."""
    p = normalize(np.asarray(pred_density, dtype=np.float64), grid_y)
    t = normalize(np.asarray(true_dens, dtype=np.float64), grid_y)
    return region_means((p - t) ** 2, grid_y, inside_sample_mask)


def hellinger_regions(pred_density, true_dens, grid_y, inside_sample_mask):
    """Return the mean squared Hellinger distance per region.

    H^2 = 1/2 * integral((sqrt(p) - sqrt(t))^2), bounded in [0, 1]; 0 = identical.
    """
    p = normalize(np.asarray(pred_density, dtype=np.float64), grid_y)
    t = normalize(np.asarray(true_dens, dtype=np.float64), grid_y)
    integrand = 0.5 * (np.sqrt(np.clip(p, 0, None)) - np.sqrt(np.clip(t, 0, None))) ** 2
    return region_means(integrand, grid_y, inside_sample_mask)


def kl_regions(pred_density, true_dens, grid_y, inside_sample_mask, eps=1e-12):
    """Return the mean forward KL divergence KL(true || pred) per region."""
    p = normalize(np.asarray(pred_density, dtype=np.float64), grid_y)
    t = normalize(np.asarray(true_dens, dtype=np.float64), grid_y)
    p = np.clip(p, eps, None)
    t = np.clip(t, eps, None)
    integrand = t * np.log(t / p)
    return region_means(integrand, grid_y, inside_sample_mask)


def point_estimates(density, grid_y):
    """Return per-row mode / mean / median of a predictive density.

    Each row is renormalized over the grid. The mean is a grid-truncated E[Y|X]
    (the population mean need not exist in the heavy-tailed regime here), so the
    median is the robust central estimate; the mode is the grid argmax.
    """
    p = normalize(np.asarray(density, dtype=np.float64), grid_y)
    grid = np.asarray(grid_y, dtype=np.float64)

    mode = grid[np.argmax(p, axis=1)]
    mean = np.trapezoid(p * grid[None, :], grid, axis=1)

    # CDF via cumulative trapezoid, normalized to end at 1, then invert at 0.5.
    dx = np.diff(grid)
    cdf = np.concatenate(
        [
            np.zeros((p.shape[0], 1)),
            np.cumsum(0.5 * (p[:, 1:] + p[:, :-1]) * dx[None, :], axis=1),
        ],
        axis=1,
    )
    cdf /= np.clip(cdf[:, -1:], 1e-12, None)
    median = np.array([np.interp(0.5, cdf[i], grid) for i in range(p.shape[0])])

    return {"mode": mode, "mean": mean, "median": median}
