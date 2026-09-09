"""Bubble-continuation vs collapse analysis for the Markov-switching bubble DGP.

Trains nothing new: it reuses the optimal MDN pipeline of ``main_simulations.py``
(Skew-t density, weighting=both, LSTM backbone) on the ``ms_bubble`` process,
MDN only. The 6k / 2k / 2k split is "train on 6k, recalibrate on 2k, evaluate
on 2k": the network is fit on the 6k train split, the LocalPIT recalibrator is fit
on the 2k calibration split, and the recalibrated predictive density is produced
on the 2k test split.

On top of that test predictive density it runs the bubble-specific study:

  1. Train 6k -> recalibrate 2k -> predict 2k  (reuses main_simulations.py machinery).
  2. Modal-clustering identification of bimodality: ToMATo (GUDHI) hill-climbs to
     the density's modes on the evaluation grid and merges any whose persistence
     falls below _MIN_PERSISTENCE (split_modes). A row left with two or more modes
     is flagged bimodal, and its mass is split at the deepest separatrix -- the
     antimode -- into P(bubble continuation) and P(collapse).
  3. Compare the MDN's mode-integrated P(continuation) to the *model-exact*
     regime probability P(S_{t+h} in {U, D} | y_{1:t}) obtained from the exact
     Hamilton filter propagated h steps through the transition matrix.

Only the MDN config.yaml must already exist (this script does NOT tune): the MDN
is trained from those params on the first run and cached per seed to
``outputs/bubble/horizon_<h>/model_seed<seed>.pt``; later runs reload it instead
of retraining (bit-exact), and a config change (e.g. n_mixtures) auto-retrains.
If the config is missing, the error message prints the exact tuning command.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from gudhi.clustering.tomato import Tomato

import main_simulations as M
from src.markov_switching_bubble.markov_switching_bubble import D, U

# All bubble-analysis artifacts live here, one subfolder per horizon.
OUTPUT_ROOT = Path("outputs/bubble")

PROCESS = "ms_bubble"
BACKBONE = "lstm"
DENSITY = "skewt"
WEIGHTING = "both"
SEED = 1

# Minimum persistence of a mode, as a fraction of the row's maximum, for it to
# survive ToMATo's merging instead of being absorbed into its neighbour.
_MIN_PERSISTENCE = 0.02

# Grid path graph, rebuilt only when the grid length changes.
_PATH_GRAPH = {}


# Step 1: load the optimal MDN and produce the recalibrated test density.
def _test_densities_path(horizon):
    """Recalibrated test densities for this horizon, one row per test case per seed."""
    return M._test_densities_path(OUTPUT_ROOT / f"horizon_{horizon}")


# Parquet contents keyed by horizon: {seed: {density, y_true, x_last_lag}}.
_DENSITY_CACHE = {}


def _load_density_cache(horizon):
    """Read this horizon's test_densities.parquet ONCE and split it by seed.

    The file is a single row group, so a per-seed filtered read decompresses all
    of it anyway; reading it once here (as main_simulations._rescore_from_cache
    does) is what keeps replay and --rescore cheap instead of paying the full
    decompression per seed.

    grid_y is deliberately not read back: it is the widest column by far -- the
    same N_GRID points repeated on every row -- and _build_eval_grid rebuilds it
    exactly from the DGP parameters. _cached_seed checks the stored density width
    against it, which is the guard the old per-seed read applied.
    """
    if horizon in _DENSITY_CACHE:
        return _DENSITY_CACHE[horizon]

    path = _test_densities_path(horizon)
    cache = {}
    if path.exists():
        df = pd.read_parquet(path, columns=["seed", "y_true", "x_last_lag", "density"])
        for s, g in df.groupby("seed"):
            cache[int(s)] = dict(
                density=np.stack(g["density"].to_numpy()).astype(np.float64),
                y_true=g["y_true"].to_numpy().astype(np.float64),
                x_last_lag=g["x_last_lag"].to_numpy().astype(np.float32),
            )
        M.logger.info(
            f"[bubble] read {path} once: {len(df)} rows, seeds {sorted(cache)}"
        )
    _DENSITY_CACHE[horizon] = cache
    return cache


def _cached_seed(horizon, seed, grid_y):
    """Return this seed's cached densities, or None if unusable."""
    entry = _load_density_cache(horizon).get(int(seed))
    if entry is None:
        return None
    if entry["density"].shape[1] != len(grid_y):
        M.logger.info(
            f"[bubble] cached densities for seed {seed} are on a "
            f"{entry['density'].shape[1]}-point grid, not {len(grid_y)}; retraining."
        )
        return None
    return entry


def _rescore_data(horizon, seed):
    """Load the minimal test split ``--rescore`` needs from the parquet.

    Only y_test and the last lag of X_test are read downstream, so the train and
    calibration tensors are never built.
    """
    entry = _load_density_cache(horizon).get(int(seed))
    if entry is None:
        raise SystemExit(
            f"--rescore: seed {seed} is not in {_test_densities_path(horizon)} -- "
            f"run without --rescore first to produce it."
        )
    return {
        "y_test": torch.as_tensor(entry["y_true"], dtype=torch.float32),
        "X_test": torch.as_tensor(entry["x_last_lag"], dtype=torch.float32)[:, None],
    }


def _fit_mdn(mdn, data, params):
    """Train the MDN on the 6k train split, validating on the 2k cali split."""
    mdn.fit(
        X_train=data["X_train"],
        y_train=data["y_train"],
        X_val=data["X_cali"],
        y_val=data["y_cali"],
        weights_train=data["weights_train"],
        weights_val=data["weights_val"],
        max_epochs=M.MAX_EPOCHS,
        learning_rate=params["learning_rate"],
        batch_size=M.BATCH_SIZE,
        max_norm=M.MAX_NORM,
        patience=M.PATIENCE,
        patience_scheduler=M.PATIENCE_SCHEDULER,
        factor_scheduler=M.FACTOR_SCHEDULER,
        weighting=M.WEIGHTING,
    )
    return mdn


def _save_test_densities(horizon, frames):
    """Append the seeds fitted in this run to test_densities.parquet."""
    path = _test_densities_path(horizon)
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = pd.concat(frames, ignore_index=True)
    if path.exists():
        old = pd.read_parquet(path)
        fresh = pd.concat(
            [old[~old["seed"].isin(fresh["seed"].unique())], fresh], ignore_index=True
        )
    # One row group per seed's worth of rows, so a later per-seed filtered read
    # can prune instead of decompressing every seed.
    fresh.to_parquet(
        path,
        index=False,
        compression="zstd",
        compression_level=9,
        row_group_size=max(fresh.groupby("seed").size().max(), 1),
    )
    M.logger.info(
        f"[bubble] recalibrated test densities cached to {path} "
        f"({len(fresh)} rows, seeds {sorted(fresh['seed'].unique().tolist())})"
    )


def _fit_and_recalibrate(horizon, seed, data, params, grid_y):
    """Train the MDN for this seed and return its recalibrated test density."""
    native = {k: M._to_native(v) for k, v in params.items()}
    M.logger.info(f"[bubble] training MDN from config: {native}")
    mdn = M.BUILDERS[BACKBONE](params, n_jobs=M.n_process)
    _fit_mdn(mdn, data, params)
    _, recal_density = M._recalibrate_test(mdn, data, grid_y)
    return np.asarray(recal_density, dtype=np.float64)


def setup(horizon):
    """Configure the run for (ms_bubble, horizon, Skew-t/both).

    Applies the main_simulations globals and loads the tuned MDN config. Call once
    before the per-seed pipeline calls.
    """
    M.init(
        horizon=horizon,
        process=PROCESS,
        density=DENSITY,
        weighting=WEIGHTING,
    )

    cfg_path = M._config_path(BACKBONE)
    if not cfg_path.exists():
        raise SystemExit(
            f"No MDN config for horizon={horizon} at {cfg_path}.\n"
            f"Provide a config.yaml (params: lags, n_mixtures, learning_rate, "
            f"dropout, lstm_depth, lstm_width) there, or produce one by tuning:\n"
            f"  python main_simulations.py --model mdn --backbone {BACKBONE} "
            f"--density {DENSITY} --weighting {WEIGHTING} --process {PROCESS} "
            f"--horizon {horizon} --n_trials <N>"
        )

    return M.load_config_yaml(BACKBONE)


def load_pipeline(horizon, seed, params, rescore=False):
    """Redraw the series under ``seed`` and return the fitted MDN and its data.

    Mirrors main_simulations._retrain_for_seed: the realization is redrawn, the
    tensor cache invalidated, then set_seed makes training deterministic.
    """
    M.build_series(PROCESS, seed=seed)  # fresh realization for this seed
    M._DATA_CACHE.clear()
    data = _rescore_data(horizon, seed) if rescore else M.get_data(params["lags"])

    q05, q95 = M._theoretical_band()  # realization-independent (DGP params only)
    grid_y = M._build_eval_grid(data, q05, q95)

    cached = _cached_seed(horizon, seed, grid_y)
    if cached is not None:
        M.logger.info(
            f"[bubble] replaying seed {seed} from {_test_densities_path(horizon)} "
            f"(no fit, no recalibration)"
        )
        recal_density = cached["density"]
        frame = None
    elif rescore:
        raise SystemExit(
            f"--rescore: seed {seed} has no usable cached density in "
            f"{_test_densities_path(horizon)} -- run without --rescore first."
        )
    else:
        M.set_seed(seed)  # deterministic init + training RNG for this seed
        recal_density = _fit_and_recalibrate(horizon, seed, data, params, grid_y)
        frame = None  # built below, once the targets are known

    # Exact closed-form true predictive density on the same grid + realized targets.
    cond_idx = M._test_conditioning_indices(data)
    true_density = np.asarray(
        M.DGP["model"].predictive_density(M.SERIES, cond_idx, horizon, grid_y),
        dtype=np.float64,
    )
    realized = data["y_test"].detach().cpu().numpy().astype(np.float64)
    x_last = data["X_test"].detach().cpu().numpy()[:, -1].astype(np.float64)

    if cached is None:
        frame = M._density_predictions_frame(
            seed, realized, x_last, grid_y, recal_density
        )

    # Drop test cases whose realized target lies outside the evaluation grid
    keep = (realized >= float(grid_y[0])) & (realized <= float(grid_y[-1]))
    n_dropped = int((~keep).sum())
    if n_dropped:
        M.logger.info(
            f"[bubble] seed {seed}: dropping {n_dropped}/{keep.size} test targets "
            f"outside the grid ({100 * n_dropped / max(keep.size, 1):.2f}%)"
        )
        recal_density = recal_density[keep]
        true_density = true_density[keep]
        realized = realized[keep]
        cond_idx = np.asarray(cond_idx)[keep]

    M.logger.info(
        f"[bubble] seed {seed} test set: n={len(cond_idx)} | grid={len(grid_y)} "
        f"points [{grid_y[0]:.2f}, {grid_y[-1]:.2f}] | "
        f"band [q05,q95]=[{q05:.2f},{q95:.2f}]"
    )
    return dict(
        seed=seed,
        horizon=horizon,
        grid_y=np.asarray(grid_y, dtype=np.float64),
        recal_density=recal_density,
        true_density=true_density,
        cond_idx=cond_idx,
        realized=realized,
        q05=q05,
        q95=q95,
        frame=frame,
    )


# Step 2: split each density at its antimode; integrate the two modes.
def _grid_path_graph(n):
    """Neighbour lists of the evaluation grid, seen as a path graph."""
    if _PATH_GRAPH.get("n") != n:
        _PATH_GRAPH.clear()
        _PATH_GRAPH["n"] = n
        _PATH_GRAPH["nbrs"] = [[max(i - 1, 0), min(i + 1, n - 1)] for i in range(n)]
    return _PATH_GRAPH["nbrs"]


def split_modes(density, grid):
    """Per row, the modal-clustering P(continuation) and the antimode splitting it."""
    n = density.shape[0]
    area = np.trapezoid(density, grid, axis=1)
    dens = density / np.where(area > 0, area, 1.0)[:, None]
    nbrs = _grid_path_graph(len(grid))

    p_cont = np.full(n, np.nan)
    antimode = np.full(n, np.nan)

    for i in range(n):
        row = dens[i]
        peak = row.max()
        if not np.isfinite(peak) or peak <= 0:
            continue

        # Persistence is expressed relative to the row's max so the threshold is
        # scale-free across test points.
        tomato = Tomato(
            graph_type="manual",
            density_type="manual",
            merge_threshold=_MIN_PERSISTENCE * peak,
        )
        tomato.fit(nbrs, weights=row)
        if tomato.n_clusters_ < 2:
            continue

        # Basin boundaries, then the deepest separatrix among them.
        labels = tomato.labels_
        cuts = np.flatnonzero(np.diff(labels) != 0)
        if not cuts.size:
            continue
        cut = int(cuts[np.argmin(np.minimum(row[cuts], row[cuts + 1]))])
        valley = cut if row[cut] <= row[cut + 1] else cut + 1

        # The two modes the separatrix divides.
        left_basin = np.flatnonzero(labels == labels[cut])
        right_basin = np.flatnonzero(labels == labels[cut + 1])
        lo = int(left_basin[np.argmax(row[left_basin])])
        hi = int(right_basin[np.argmax(row[right_basin])])
        antimode[i] = grid[valley]

        left = np.trapezoid(row[: valley + 1], grid[: valley + 1])
        right = np.trapezoid(row[valley:], grid[valley:])
        # Continuation = far-from-zero peak's side; collapse = inner peak's side.
        p_cont[i] = right if abs(grid[hi]) >= abs(grid[lo]) else left
    return p_cont, antimode


# Step 4: model-exact regime probabilities from the Hamilton filter.
def exact_regime_probabilities(model, series, cond_idx, horizon):
    """Model-exact P(S_{t+h} = k | y_{1:t}) per test point.

    Runs the exact Hamilton filter on the full path, then propagates the filtered
    regime distribution h steps through the transition matrix. Continuation =
    P(bubble regime U or D at t+h).
    """
    xi = model.filter_probabilities(series)  # (N, 3)
    # np.errstate guards a spurious macOS Accelerate-BLAS FP warning on this
    # benign probability matmul (inputs are all in [0, 1]; output verified valid).
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        P_h = np.linalg.matrix_power(model.P, horizon)
        xi_h = xi[cond_idx] @ P_h
    return xi_h[:, U] + xi_h[:, D]


# Reporting: summary metrics.
def _detection_stats(bim_mdn, bim_true):
    """Confusion counts and recall/precision/F1 for the MDN bimodality flags.

    Recall is the share of truly bimodal densities the MDN also flags, precision the
    share of its flags that are real. Both are NaN when undefined.
    """
    bim_mdn = np.asarray(bim_mdn, dtype=bool)
    bim_true = np.asarray(bim_true, dtype=bool)
    hits = int(np.sum(bim_mdn & bim_true))
    n_true = int(np.sum(bim_true))
    n_mdn = int(np.sum(bim_mdn))
    recall = hits / n_true if n_true else float("nan")
    precision = hits / n_mdn if n_mdn else float("nan")
    denom = (precision + recall) if (n_true and n_mdn) else 0.0
    f1 = (2 * precision * recall / denom) if denom > 0 else float("nan")
    return {
        "n_bimodal_true": n_true,
        "n_bimodal_mdn": n_mdn,
        "recall_recovery": recall,
        "precision": precision,
        "f1": f1,
    }


def analyze_seed(res):
    """Run steps 2-3 for a single seed and return the summary metrics.

    The continuation-magnitude metrics are always present, NaN when fewer than
    five in-bubble points are recovered.
    """
    grid = res["grid_y"]
    fhat = res["recal_density"]
    ftrue = res["true_density"]
    horizon = res["horizon"]
    model = M.DGP["model"]

    # Step 2: modal-clustering identification + continuation/collapse split
    pc_mdn, antimode_mdn = split_modes(fhat, grid)
    _, antimode_true = split_modes(ftrue, grid)
    is_bimodal = np.isfinite(antimode_mdn)  # MDN flagged bimodal
    is_bimodal_true = np.isfinite(antimode_true)  # true density bimodal

    # Step 3: model-exact regime probabilities.
    pc_exact = exact_regime_probabilities(model, M.SERIES, res["cond_idx"], horizon)

    # Bimodality detection: MDN vs true (how many, and how much recovered)
    detection = _detection_stats(is_bimodal, is_bimodal_true)

    # Continuation magnitude on the identified in-bubble subset
    survive = float((1.0 - model.hazard) ** horizon)  # exact bubble continuation
    true_bubble = pc_exact > 0.5
    recovered = true_bubble & np.isfinite(pc_mdn)  # identified bimodal & in bubble
    n_true_bubble = int(true_bubble.sum())
    n_recovered = int(recovered.sum())

    summary = {
        "horizon": horizon,
        "detection": detection,
        "p_cont_survive": survive,
        "n_true_bubble": n_true_bubble,
        "recovery_rate_bubble": (
            n_recovered / n_true_bubble if n_true_bubble else float("nan")
        ),
    }
    if n_recovered >= 5:
        pred, true = pc_mdn[recovered], pc_exact[recovered]
        summary["bias_cont_bubble"] = float(np.mean(pred - true))
    else:
        M.logger.warning(
            f"[bubble] seed {res['seed']}: only {n_recovered} recovered true-bubble "
            f"points; continuation-magnitude metrics are NaN here (need >=5)."
        )
        summary["bias_cont_bubble"] = float("nan")

    return summary


def _prune_nan(node):
    """Drop NaN scalars, and any dict left empty by that, from a summary tree."""
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            v = _prune_nan(v)
            if v is None:
                continue
            if isinstance(v, dict) and not v:
                continue
            out[k] = v
        return out
    if isinstance(node, float) and math.isnan(node):
        return None
    return node


def make_reports(horizon, params, out_dir, rescore=False):
    """Evaluate the bubble study and write the summary."""
    what = "re-score" if rescore else "fit + evaluate"
    M.logger.info(f"[bubble] seed {SEED}: {what}")
    res = load_pipeline(horizon, SEED, params, rescore=rescore)
    if res["frame"] is not None:
        _save_test_densities(horizon, [res["frame"]])
    summary = analyze_seed(res)

    M.logger.info(
        f"[bubble] summary (h={horizon}): "
        f"recovery_rate_bubble={summary['recovery_rate_bubble']:.3f} | "
        f"detection F1={summary['detection']['f1']:.3f}"
    )
    json_path = out_dir / "bubble_summary.json"
    with open(json_path, "w") as f:
        json.dump(_prune_nan(summary), f, indent=2)
    M.logger.info(f"[bubble] summary -> {json_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--horizon",
        type=int,
        required=True,
        help="Forecast horizon h (predict Y_{t+h} from the path up to t).",
    )
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="Re-score the cached test densities without fitting anything.",
    )
    parser.add_argument(
        "--persistence",
        type=float,
        default=None,
        help=f"Override _MIN_PERSISTENCE (default {_MIN_PERSISTENCE}): the drop from "
        f"a mode's peak to the saddle joining it to a higher one, as a fraction of "
        f"the row's max, below which ToMATo merges the two.",
    )
    args = parser.parse_args()

    if args.persistence is not None:
        _MIN_PERSISTENCE = args.persistence
    M.logger.info(f"[bubble] min persistence = {_MIN_PERSISTENCE}")

    params = setup(args.horizon)
    out_dir = OUTPUT_ROOT / f"horizon_{args.horizon}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.rescore and SEED not in _load_density_cache(args.horizon):
        raise SystemExit(
            f"--rescore: {_test_densities_path(args.horizon)} holds no seed "
            f"{SEED} (run without --rescore first to produce it)."
        )

    make_reports(args.horizon, params, out_dir, rescore=args.rescore)
    M.logger.info(f"[bubble] done. Outputs under {out_dir}")
