"""Prediction-region metrics on the highest-density region (HDR).

The (1 - alpha) HDR is the super-level set {y : p(y) >= c} of that mass: one
interval for a unimodal density, a union of intervals for a multimodal one.
Every method uses the same construction.
"""

import numpy as np


def hdr_region_from_row(dens, grid, alpha=0.05):
    """Return the highest-density (1 - alpha) region from one density row.

    The row is renormalized; the threshold c is chosen so {y : p(y) >= c} has
    mass (1 - alpha), and the super-level set is returned as a list of contiguous
    (lower, upper) intervals (one for a unimodal row, several for a multimodal one).
    """
    dens = np.clip(np.asarray(dens, dtype=np.float64), 0.0, None)
    grid = np.asarray(grid, dtype=np.float64)

    mass = np.trapezoid(dens, grid)
    if mass <= 1e-12:  # degenerate density: fall back to full support
        return [(float(grid[0]), float(grid[-1]))]

    p = dens / mass

    # Per-cell mass; accumulate from the highest density down to (1 - alpha) and
    # read off the threshold density at the cutoff cell.
    widths = np.gradient(grid)
    cell_mass = p * widths
    order = np.argsort(p)[::-1]  # densities, highest first
    cum = np.cumsum(cell_mass[order])
    cum /= max(cum[-1], 1e-12)
    k = int(np.searchsorted(cum, 1.0 - alpha))
    k = min(k, len(order) - 1)
    threshold = p[order[k]]

    included = p >= threshold

    # Contiguous runs of the super-level set -> intervals.
    intervals = []
    i, n = 0, len(grid)
    while i < n:
        if included[i]:
            j = i
            while j + 1 < n and included[j + 1]:
                j += 1
            intervals.append((float(grid[i]), float(grid[j])))
            i = j + 1
        else:
            i += 1
    return intervals or [(float(grid[0]), float(grid[-1]))]


def hdr_regions(density, grid_y, alpha=0.05):
    """Return per-sample HDR regions for a (n_samples, n_grid) predictive density."""
    density = np.asarray(density, dtype=np.float64)
    return [
        hdr_region_from_row(density[i], grid_y, alpha) for i in range(density.shape[0])
    ]


def coverage_regions(regions, y_true, inside_sample_mask):
    """Return the empirical coverage of the prediction region per sample region."""
    y = np.asarray(y_true, dtype=np.float64)
    covered = np.array(
        [any(lo <= y[i] <= hi for lo, hi in regions[i]) for i in range(len(regions))]
    )
    inside = covered[inside_sample_mask]
    outside = covered[~inside_sample_mask]
    return {
        "all": float(covered.mean()),
        "inside": float(inside.mean()) if inside.size else float("nan"),
        "outside": float(outside.mean()) if outside.size else float("nan"),
    }


def length_regions(regions, inside_sample_mask):
    """Return the mean total prediction-region length per sample region."""
    width = np.array([sum(hi - lo for lo, hi in r) for r in regions], dtype=np.float64)
    inside = width[inside_sample_mask]
    outside = width[~inside_sample_mask]
    return {
        "all": float(width.mean()),
        "inside": float(inside.mean()) if inside.size else float("nan"),
        "outside": float(outside.mean()) if outside.size else float("nan"),
    }


def winkler_regions(regions, y_true, inside_sample_mask, alpha=0.05):
    """Return the mean Winkler / interval score over the HDR's disjoint intervals."""
    y = np.asarray(y_true, dtype=np.float64)
    score = np.empty(len(regions), dtype=np.float64)
    for i, r in enumerate(regions):
        total_width = sum(hi - lo for lo, hi in r)
        if any(lo <= y[i] <= hi for lo, hi in r):
            dist = 0.0
        else:
            dist = min(min(abs(y[i] - lo), abs(y[i] - hi)) for lo, hi in r)
        score[i] = total_width + (2.0 / alpha) * dist
    inside = score[inside_sample_mask]
    outside = score[~inside_sample_mask]
    return {
        "all": float(score.mean()),
        "inside": float(inside.mean()) if inside.size else float("nan"),
        "outside": float(outside.mean()) if outside.size else float("nan"),
    }


def interval_metrics(regions, y_test_np, inside_sample_mask, alpha):
    """Return coverage / length / Winkler for one set of regions."""
    return {
        "coverage": coverage_regions(regions, y_test_np, inside_sample_mask),
        "ci_length": length_regions(regions, inside_sample_mask),
        "winkler": winkler_regions(regions, y_test_np, inside_sample_mask, alpha=alpha),
    }
