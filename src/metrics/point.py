"""Point-prediction MSE/MAE, partitioned by the realized target.

The density metrics partition the grid instead.
"""

import numpy as np


def mse_regions(point_pred, y_true, inside_sample_mask):
    """Return per-region MSE of a scalar point prediction vs the realized target.

    Partitions the test SAMPLES (not the grid) by where the realized target lands
    relative to the theoretical band: 'inside' = [q05, q95], 'outside' = tails,
    'all' = every sample.
    """
    se = (np.asarray(point_pred, dtype=np.float64) - np.asarray(y_true)) ** 2
    inside = se[inside_sample_mask]
    outside = se[~inside_sample_mask]
    return {
        "all": float(se.mean()),
        "inside": float(inside.mean()) if inside.size else float("nan"),
        "outside": float(outside.mean()) if outside.size else float("nan"),
    }


def mae_regions(point_pred, y_true, inside_sample_mask):
    """Return per-region MAE of a scalar point prediction vs the realized target.

    Same sample-region partition as mse_regions. MAE is the natural companion
    metric in the heavy-tailed regime, where the population MSE need not exist
    and the empirical MSE is dominated by the few most extreme targets.
    """
    ae = np.abs(np.asarray(point_pred, dtype=np.float64) - np.asarray(y_true))
    inside = ae[inside_sample_mask]
    outside = ae[~inside_sample_mask]
    return {
        "all": float(ae.mean()),
        "inside": float(inside.mean()) if inside.size else float("nan"),
        "outside": float(outside.mean()) if outside.size else float("nan"),
    }


def no_change_mse(data, y_true, inside_sample_mask):
    """Return the no-change (random-walk) benchmark MSE over the sample regions."""
    no_change_pred = data["X_test"].detach().cpu().numpy()[:, -1].astype(np.float64)
    return mse_regions(no_change_pred, y_true, inside_sample_mask)


def no_change_mae(data, y_true, inside_sample_mask):
    """Return the no-change (random-walk) benchmark MAE over the sample regions."""
    no_change_pred = data["X_test"].detach().cpu().numpy()[:, -1].astype(np.float64)
    return mae_regions(no_change_pred, y_true, inside_sample_mask)


def relative_mse(method_mse, no_change_mse):
    """Return the per-region relative MSE (method / no-change).

    Paired within a region so the ratio is formed within one realization before
    averaging across seeds. A region is NaN if either MSE is NaN or the
    benchmark MSE is non-positive.
    """
    out = {}
    for region in ("all", "inside", "outside"):
        num = method_mse[region]
        den = no_change_mse[region]
        if np.isfinite(num) and np.isfinite(den) and den > 0.0:
            out[region] = float(num / den)
        else:
            out[region] = float("nan")
    return out


# The paired per-region ratio is metric-agnostic; reuse it for MAE.
relative_mae = relative_mse
