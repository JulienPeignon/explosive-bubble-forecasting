"""Writers and formatters for the per-run artifacts.

JSON metrics, YAML configs, the density parquets and the Optuna trials tracker.
Pure I/O, so every forecast method can share them.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.results.paths import (
    pit_calibration_path,
    point_predictions_path,
    raw_densities_path,
    test_densities_path,
    trials_path,
)
from src.utils.setup_logger import setup_logger

logger = setup_logger()


def format_number_4_digits(value):
    """Format a number to four significant digits."""
    if value is None or value == "":
        return ""
    if abs(value) < 0.001:
        return f"{value:.3e}"
    abs_val = abs(value)
    if abs_val >= 10000:
        return f"{value:.3e}"
    elif abs_val >= 1000:
        return f"{value:.0f}"
    elif abs_val >= 100:
        return f"{value:.1f}"
    elif abs_val >= 10:
        return f"{value:.2f}"
    elif abs_val >= 1:
        return f"{value:.3f}"
    else:
        return f"{value:.3f}"


def round_floats(obj):
    """Round every float in a nested structure for JSON output."""
    if isinstance(obj, float):
        return format_number_4_digits(obj)
    if isinstance(obj, dict):
        return {k: round_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_floats(v) for v in obj]
    return obj


def to_native(v):
    """Convert numpy / torch scalars to plain Python types for clean YAML."""
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def write_json(path, payload):
    """Round floats to 4 significant digits and dump as indented JSON."""
    with open(path, "w") as f:
        json.dump(round_floats(payload), f, indent=2)


def write_yaml(path, cfg):
    """Write ``cfg`` as YAML."""
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    return path


def load_trials(out_dir):
    """Return the trials tracker, or an empty dict if it does not exist yet."""
    p = trials_path(out_dir)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


def save_trials(out_dir, tracker):
    """Persist the trials tracker to disk and log its contents."""
    p = trials_path(out_dir)
    with open(p, "w") as f:
        json.dump(tracker, f, indent=2)
    summary = " | ".join(f"{b}={n}" for b, n in sorted(tracker.items()))
    logger.info(f"Trials tracker updated → {p}  [{summary}]")


def save_model_weights(path, model, params):
    """Persist a fitted model's weights and fitted scalers, keyed by its params."""
    torch.save(
        {
            "params": params,
            "state_dict": model.state_dict(),
            "scaler_x": getattr(model, "scaler_x", None),
            "scaler_y": getattr(model, "scaler_y", None),
        },
        path,
    )
    logger.info(f"model weights saved to {path}")


def load_model_weights(path, model, params):
    """Restore weights and scalers into ``model``.

    False when the checkpoint is for different params, so the caller retrains.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("params") != params:
        logger.warning(f"{path} was written for different params -- ignoring it")
        return False
    model.load_state_dict(ckpt["state_dict"])
    if ckpt.get("scaler_x") is not None:
        model.scaler_x = ckpt["scaler_x"]
    if ckpt.get("scaler_y") is not None:
        model.scaler_y = ckpt["scaler_y"]
    return True


def density_predictions_frame(
    seed, y_true, x_last_lag, grid_y, density, pred_grid_size, x_test=None
):
    """Return one seed's uncorrected test densities as a long-format DataFrame.

    Subsampled onto `pred_grid_size` grid points to bound parquet size.
    `x_last_lag` is Y_{t-h}, kept so the closed-form density / no-change
    benchmark can be recomputed without rerunning the model. `x_test` is the
    full conditioning vector, which the recalibrator needs to be re-applied --
    see `calibration_frame` for the other half.
    """
    idx = np.linspace(0, len(grid_y) - 1, pred_grid_size).round().astype(int)
    grid_small = np.asarray(grid_y, dtype=np.float32)[idx]
    density_small = np.asarray(density, dtype=np.float32)[:, idx]
    n = density_small.shape[0]
    return pd.DataFrame(
        {
            "seed": np.full(n, seed, dtype=np.int32),
            "sample_idx": np.arange(n, dtype=np.int32),
            "y_true": np.asarray(y_true, dtype=np.float32),
            "x_last_lag": np.asarray(x_last_lag, dtype=np.float32),
            "grid_y": [grid_small] * n,
            "density": list(density_small),
            **(
                {"x_test": list(np.asarray(x_test, dtype=np.float32))}
                if x_test is not None
                else {}
            ),
        }
    )


def point_predictions_frame(seed, y_true, x_last_lag, pred):
    """Return one seed's point forecasts as a DataFrame."""
    pred = np.asarray(pred, dtype=np.float64).ravel()
    n = len(pred)
    return pd.DataFrame(
        {
            "seed": np.full(n, seed, dtype=np.int32),
            "sample_idx": np.arange(n, dtype=np.int32),
            "y_true": np.asarray(y_true, dtype=np.float32).ravel(),
            "x_last_lag": np.asarray(x_last_lag, dtype=np.float32).ravel(),
            "pred": pred.astype(np.float32),
        }
    )


def save_point_parquet(out_dir, frames):
    """Write the point-forecast artifact of one run; return the pooled frame."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = _pooled(frames)
    df.to_parquet(point_predictions_path(out_dir), index=False, **_PARQUET_KWARGS)
    logger.info(
        f"point forecasts saved to {point_predictions_path(out_dir)} ({len(df)} rows)"
    )
    return df


def calibration_frame(seed, pit_cali, x_cali):
    """Return one seed's calibration PITs and features as a DataFrame.

    These are everything the LocalPITRecalibrator is fitted on, so keeping them
    lets the correction be redone at another `num_basis` -- or with different
    calibrator settings entirely -- without refitting the model.
    """
    pit = np.asarray(pit_cali, dtype=np.float32).ravel()
    x = np.asarray(x_cali, dtype=np.float32)
    n = len(pit)
    return pd.DataFrame(
        {
            "seed": np.full(n, seed, dtype=np.int32),
            "cali_idx": np.arange(n, dtype=np.int32),
            "pit": pit,
            "x_cali": list(x),
        }
    )


_PARQUET_KWARGS = {"compression": "zstd", "compression_level": 9}


def _pooled(frames):
    """One DataFrame from either a single frame or a list of per-seed frames."""
    if isinstance(frames, pd.DataFrame):
        return frames
    return pd.concat(frames, ignore_index=True)


def save_density_parquets(out_dir, raw_frames, test_frames, cali_frames):
    """Write the density artifacts of one run; return their pooled frames."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    test, cali = _pooled(test_frames), _pooled(cali_frames)

    raw = None
    if raw_frames is not None:
        raw = _pooled(raw_frames)
        raw.to_parquet(raw_densities_path(out_dir), index=False, **_PARQUET_KWARGS)
    test.to_parquet(test_densities_path(out_dir), index=False, **_PARQUET_KWARGS)
    cali.to_parquet(pit_calibration_path(out_dir), index=False, **_PARQUET_KWARGS)
    logger.info(
        f"densities saved to {out_dir} "
        f"(raw={len(raw) if raw is not None else 'not saved'}, "
        f"test={len(test)} rows, pit={len(cali)} rows)"
    )

    return raw, test, cali
