"""Aggregation, persistence and printing of multi-seed evaluation results.

Pure formatting and I/O: the only run state, the process name, is passed in, so
every forecast method can share this layer.
"""

import numpy as np

from src.results.io import write_json
from src.utils.setup_logger import setup_logger

logger = setup_logger()

# Sample/grid regions every metric is reported over, in table-column order.
REGIONS = ("all", "inside", "outside")


# Per-seed aggregation
def aggregate_over_seeds(per_seed):
    """Aggregate a list of per-seed metric dicts into mean / std / raw values.

    Returns metric -> region -> {"mean", "std", "values"}. std is the sample std
    (ddof=1) over the finite seed values; NaN if fewer than two are finite. NaNs
    are skipped (e.g. an empty 'outside' region in some seed). Regions are taken
    from the metric itself, so total-only metrics (the pinball scores) stay
    total-only.
    """
    agg = {}
    for metric in per_seed[0]:
        agg[metric] = {}
        for region in per_seed[0][metric]:
            vals = np.array([d[metric][region] for d in per_seed], dtype=np.float64)
            finite = vals[np.isfinite(vals)]
            agg[metric][region] = {
                "mean": float(finite.mean()) if finite.size else float("nan"),
                "std": float(finite.std(ddof=1)) if finite.size > 1 else float("nan"),
                "values": [float(v) for v in vals],
            }
    return agg


def drop_seed_values(agg):
    """Return the aggregated metrics without the per-seed ``values`` arrays."""
    return {
        metric: {
            region: {k: v for k, v in cell.items() if k != "values"}
            for region, cell in regions.items()
        }
        for metric, regions in agg.items()
    }


_PERSISTED_INTERVAL_KEYS = (
    "coverage",
    "ci_length",
    "winkler",
    "coverage_eqt",
    "ci_length_eqt",
    "winkler_eqt",
    "coverage_short",
    "ci_length_short",
    "winkler_short",
)
_PERSISTED_DENSITY_KEYS = (
    "ise",
    "kl",
    "hellinger",
    "pinball_q01",
    "pinball_q05",
    "pinball_q10",
    "pinball_q90",
    "pinball_q95",
    "pinball_q99",
)
_PERSISTED_POINT_KEYS = (
    "mse_median_rel",
    "mse_mode_rel",
    "mse_mean_rel",
    "mae_median_rel",
    "mae_mode_rel",
    "mae_mean_rel",
)


def persisted_metrics(metrics, *, point=False):
    """Drop metric keys no output table consumes.

    Interval and density keys are kept for every method; the relative point-MSE
    keys only when ``point``. Filtering is by intersection, so closed-form-only
    keys pass through when present.
    """
    keep = set(_PERSISTED_INTERVAL_KEYS) | set(_PERSISTED_DENSITY_KEYS)
    if point:
        keep |= set(_PERSISTED_POINT_KEYS)
    return {k: v for k, v in metrics.items() if k in keep}


def write_multiseed_eval_jsons(
    out_dir,
    *,
    process_name,
    json_key,
    model_label,
    lags,
    seeds,
    agg,
    log_tag,
    point=False,
    extra=None,
):
    """Write the test_eval.json / test_eval_detailed.json pair.

    The first holds the persisted subset as mean +/- std, the second every
    metric including per-seed values. ``extra`` is merged in after ``lags``.
    """
    extra = extra or {}
    write_json(
        out_dir / "test_eval.json",
        {
            json_key: model_label,
            "process": process_name,
            "lags": lags,
            **extra,
            "n_seeds": len(seeds),
            "metrics": persisted_metrics(drop_seed_values(agg), point=point),
        },
    )
    write_json(
        out_dir / "test_eval_detailed.json",
        {
            json_key: model_label,
            "process": process_name,
            "lags": lags,
            **extra,
            "seeds": list(seeds),
            "metrics": agg,
        },
    )
    logger.info(
        f"{log_tag} aggregated test metrics over {len(seeds)} seeds "
        f"saved to {out_dir}/test_eval.json and test_eval_detailed.json"
    )


# Comparison tables
def fmt_cell(cell):
    """Format an aggregated {'mean','std',...} region entry as 'mean+/-std'."""
    mean, std = cell["mean"], cell["std"]
    if not np.isfinite(mean):
        return "n/a"
    if not np.isfinite(std):
        return f"{mean:.4f}"
    return f"{mean:.4f}+/-{std:.4f}"


# Metric labels shared by every comparison table.
_METRIC_TITLES = {
    "coverage": "Coverage of the 95% HDR (target 0.95)",
    "ci_length": "Mean total length of the 95% HDR",
    "winkler": "Winkler interval score, 95% HDR (lower is better)",
    "mse_mode_rel": "Relative MSE -- conditional mode / no-change",
    "mse_mean_rel": "Relative MSE -- conditional expectation / no-change",
    "mse_median_rel": "Relative MSE -- conditional median / no-change",
    "mae_mode_rel": "Relative MAE -- conditional mode / no-change",
    "mae_mean_rel": "Relative MAE -- conditional expectation / no-change",
    "mae_median_rel": "Relative MAE -- conditional median / no-change",
    "ise": "ISE (Integrated Squared Error)",
    "hellinger": "Squared Hellinger distance H^2",
    "kl": "KL divergence  KL(true || pred)",
    "pinball_q01": "Pinball loss, quantile 0.01",
    "pinball_q05": "Pinball loss, quantile 0.05",
    "pinball_q10": "Pinball loss, quantile 0.10",
    "pinball_q90": "Pinball loss, quantile 0.90",
    "pinball_q95": "Pinball loss, quantile 0.95",
    "pinball_q99": "Pinball loss, quantile 0.99",
    "coverage_eqt": "Coverage of the 95% equal-tailed interval (target 0.95)",
    "ci_length_eqt": "Mean length of the 95% equal-tailed interval",
    "winkler_eqt": "Winkler score, 95% equal-tailed interval (lower is better)",
    "coverage_short": "Coverage of the 95% shortest interval (target 0.95)",
    "ci_length_short": "Mean length of the 95% shortest interval",
    "winkler_short": "Winkler score, 95% shortest interval (lower is better)",
}
_METRIC_CAPTIONS = {
    "coverage": "realized target in the HDR; regions partition the samples",
    "ci_length": "total width of the HDR; regions partition the samples",
    "winkler": "interval score vs realized target; regions partition the samples",
    "mse_mode_rel": "<1 beats no-change; per-seed ratio",
    "mse_mean_rel": "<1 beats no-change; per-seed ratio",
    "mse_median_rel": "<1 beats no-change; per-seed ratio",
    "mae_mode_rel": "<1 beats no-change; per-seed ratio",
    "mae_mean_rel": "<1 beats no-change; per-seed ratio",
    "mae_median_rel": "<1 beats no-change; per-seed ratio",
    "ise": "recalibrated vs true density; regions partition the samples",
    "hellinger": "recalibrated vs true density; regions partition the samples",
    "kl": "recalibrated vs true density; regions partition the samples",
    "pinball_q01": "quantile loss vs realized target; whole test set",
    "pinball_q05": "quantile loss vs realized target; whole test set",
    "pinball_q10": "quantile loss vs realized target; whole test set",
    "pinball_q90": "quantile loss vs realized target; whole test set",
    "pinball_q95": "quantile loss vs realized target; whole test set",
    "pinball_q99": "quantile loss vs realized target; whole test set",
    "coverage_eqt": "realized target in the interval; regions partition the samples",
    "ci_length_eqt": "interval width; regions partition the samples",
    "winkler_eqt": "interval score vs realized target; regions partition the samples",
    "coverage_short": "realized target in the interval; regions partition the samples",
    "ci_length_short": "interval width; regions partition the samples",
    "winkler_short": "interval score vs realized target; regions partition the samples",
}


def print_eval_tables(eval_results, seeds, label_header="backbone"):
    """Print one comparison table per metric for a {label: aggregated-metrics} dict.

    Each cell is mean+/-std over `seeds`; only metrics present in the results are
    printed (e.g. ISE/Hellinger/KL appear only for closed-form DGPs).
    """
    present = set(next(iter(eval_results.values())))
    cw = 18  # column width for "mean+/-std" cells
    header = (
        f"{label_header:<12} | {'all':>{cw}} | {'inside[5-95%]':>{cw}} | "
        f"{'outside':>{cw}}"
    )
    for metric, title in _METRIC_TITLES.items():
        if metric not in present:
            continue
        logger.info("\n" + "=" * len(header))
        logger.info(
            f"{title} ({_METRIC_CAPTIONS[metric]}) "
            f"-- mean+/-std over {len(seeds)} seeds {list(seeds)}"
        )
        logger.info("=" * len(header))
        logger.info(header)
        logger.info("-" * len(header))
        for label, res in eval_results.items():
            m = res[metric]
            cells = [fmt_cell(m[r]) if r in m else "" for r in REGIONS]
            logger.info(f"{label:<12} | " + " | ".join(f"{c:>{cw}}" for c in cells))
        logger.info("=" * len(header))
