"""CDF and quantile metrics: pinball loss and interval constructions.

The equal-tailed and shortest-connected intervals are alternatives to the HDR
in :mod:`src.metrics.regions`.
"""

import numpy as np

from src.metrics.common import normalize

PINBALL_LEVELS = (0.01, 0.05, 0.10, 0.90, 0.95, 0.99)


def cdf_from_density(density, grid_y):
    """Return the per-row CDF via cumulative trapezoid of the renormalized density."""
    p = normalize(np.asarray(density, dtype=np.float64), grid_y)
    grid = np.asarray(grid_y, dtype=np.float64)
    dx = np.diff(grid)
    cdf = np.concatenate(
        [
            np.zeros((p.shape[0], 1)),
            np.cumsum(0.5 * (p[:, 1:] + p[:, :-1]) * dx[None, :], axis=1),
        ],
        axis=1,
    )
    cdf /= np.clip(cdf[:, -1:], 1e-12, None)
    return cdf


def quantiles_from_cdf(cdf, grid_y, levels):
    """Return quantiles at `levels` by interpolating the grid against each row's CDF."""
    grid = np.asarray(grid_y, dtype=np.float64)
    levels = np.atleast_1d(np.asarray(levels, dtype=np.float64))
    out = np.empty((cdf.shape[0], levels.size), dtype=np.float64)
    for i in range(cdf.shape[0]):
        out[i] = np.interp(levels, cdf[i], grid)
    return out


def sample_total_mean(values):
    """Return the mean of a per-sample scalar over the whole test set only.

    For the pinball score: the bulk/tail split partitions the samples by where
    the realization landed, which is the very thing a quantile at level `level`
    is being scored on -- so only the total is meaningful.
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    return {"all": float(v.mean()) if v.size else float("nan")}


def pinball_samples(q_pred, y_true, level):
    """Return the per-sample pinball / quantile loss at `level`."""
    q = np.asarray(q_pred, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.float64)
    return np.where(y >= q, level * (y - q), (1.0 - level) * (q - y))


def equal_tailed_regions(cdf, grid_y, alpha=0.05):
    """Return the equal-tailed (1 - alpha) interval per row, in HDR format."""
    q = quantiles_from_cdf(cdf, grid_y, [alpha / 2.0, 1.0 - alpha / 2.0])
    return [[(float(q[i, 0]), float(q[i, 1]))] for i in range(q.shape[0])]


def shortest_regions(cdf, grid_y, alpha=0.05, n_p=51):
    """Return the shortest connected (1 - alpha) interval per row.

    Slides the lower-tail mass p over [0, alpha] and keeps the
    [F^-1(p), F^-1(p + 1 - alpha)] pair of minimal width, in the HDR region
    format. For a multimodal density the true HDR may be a shorter disjoint set.
    """
    p_grid = np.linspace(0.0, alpha, n_p)
    lo = quantiles_from_cdf(cdf, grid_y, p_grid)
    hi = quantiles_from_cdf(cdf, grid_y, p_grid + (1.0 - alpha))
    j = np.argmin(hi - lo, axis=1)
    return [[(float(lo[i, j[i]]), float(hi[i, j[i]]))] for i in range(lo.shape[0])]
