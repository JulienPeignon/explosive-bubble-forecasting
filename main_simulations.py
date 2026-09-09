# ruff: noqa: E402
"""
Optuna hyperparameter search.

======================================================================
COMMON PARAMETERS  (shared across modes)
======================================================================

    lags            int   [1, 10]                 # input length
    learning_rate   cat   {1e-4, 2e-4, 4e-4, 6e-4, 8e-4,
                           1e-3, 2e-3, 4e-3, 6e-3, 8e-3, 1e-2}   # Adam LR
    dropout         float {0.0, 0.05, 0.10, 0.15, 0.20}  # step 0.05

    # LSTM: between-layer + pooled-feature dropout

    depth       int   [1, 5]                     # number of hidden layers
    width       cat   {64, 128, 256, 512}        # width of every layer
                                                 # -> hidden_layers = [width] * depth

    # (last entry = feature width to MDN heads)

======================================================================
PARAMETERS BEING OPTIMIZED  (per model / CLI mode)
======================================================================

# MDN
#          Headline skew-t model (weighting=both) + density/weighting
#          ablations.   Objective: validation NLL.

SHARED: lags, learning_rate, dropout (see COMMON PARAMETERS)
LSTM: lstm_depth, lstm_width (see COMMON PARAMETERS)

    n_mixtures      int   [1, 10]                # MDN density resolution

FIXED (NOT searched)
    batch_size=512, point_head_layers=[32]
    max_epochs=1000, max_norm=30.0, patience=30, patience_scheduler=10,
    factor_scheduler=0.5
    (point_head_layers has no effect on the NLL, since fit() excludes point_head
     from the density optimizer.)

# POINT HEAD  --  --model point_head  (only with --density skewt)
#                 Regression head on top of the FROZEN density backbone
#                 (trained from the density config).  Objective: validation MSE.

    ph_depth         int  [0, 2]                  # number of point-head hidden layers
    ph_width         cat  {16, 32, 64}            # width of every point-head layer;
                                                  # only suggested when ph_depth > 0

    # -> point_head_layers = [ph_width] * ph_depth

    # (depth 0 => bare linear head, no width)
    ph_learning_rate cat  (same grid as learning_rate; see COMMON PARAMETERS)

FIXED (NOT searched)
    density backbone + heads (lags, n_mixtures, width, ...) frozen from the
    density config
    batch_size=512, loss=MSE
    max_epochs=1000, patience=30, patience_scheduler=10, factor_scheduler=0.5

# FLEXZBOOST  --  --model flexzboost
#                 FlexCode CDE with an XGBoost basis-coefficient regressor.
#                 Objective: validation CDE loss.

SHARED: lags (see COMMON PARAMETERS)

    basis_system    cat   {cosine, Fourier, db4}  # FlexCode expansion basis
    n_estimators    int   [100, 2000] log-uniform  # XGBoost trees
    max_depth       int   [2, 10] step 2          # XGBoost tree depth
    learning_rate   cat   {6e-3, 8e-3, 1e-2, 2e-2, 4e-2,
                           6e-2, 8e-2, 1e-1, 2e-1, 4e-1, 6e-1}
    # XGBoost LR (fixed grid,

    # distinct from the Adam LR

    # grid in COMMON PARAMETERS)

FIXED (NOT searched by Optuna)
    max_basis=40 (FLEXZBOOST_MAX_BASIS); FlexCode .tune() selects the cutoff
    xgb objective=reg:squarederror, n_grid
    (.tune() also selects, on the cali set:
       bump_threshold in {linspace(0.0, 0.2, 9)}
       sharpen        in {linspace(0.5, 2.0, 7)})

# KCDE  --  --model kcde
#           Direct kernel CDE with independent conditioning/response
#           bandwidths.   Objective: validation CDE NLL.

SHARED: lags (see COMMON PARAMETERS)

    bandwidth_x     cat   {0.10, 0.15, 0.20, 0.30, 0.50,
                           0.75, 1.00, 1.50, 2.00}
    # conditioning (lag) product-kernel bw
    bandwidth_y     cat   {0.10, 0.15, 0.20, 0.30, 0.50,
                           0.75, 1.00, 1.50, 2.00}  # response-kernel bw (same grid)
    kernel          cat   {gaussian, epanechnikov, tricube}  # kernel family

FIXED (NOT searched)
    scaler=RobustScaler (fit on train only), n_grid, chunk_size=512

# FLOW  --  --model flow
#           Conditional NSF (rational-quadratic neural spline), UNWEIGHTED
#           maximum likelihood.  Objective: val NLL.

SHARED: lags, learning_rate (see COMMON PARAMETERS)

    transforms      int   [1, 5]                  # number of spline transforms

    # (density-resolution axis ~ n_mixtures)
    hidden_depth    int   [1, 5]                  # conditioner network depth
    hidden_width    cat   {64, 128, 256, 512}     # conditioner network width
    bins            cat   {4, 8, 16}              # rational-quadratic spline bins

FIXED (NOT searched)
    features=1 (scalar target), base_distribution=standard_normal
    scaler=RobustScaler, weighting=none, n_grid
    batch_size=512, max_norm=30.0, max_epochs=1000, patience=30,
    patience_scheduler=10, factor_scheduler=0.5
    (no dropout: zuko's conditioner has none; transforms/bins control capacity)

# CONFORMAL  --  --model conformal
#                Unweighted point-predictor backbone (fit_mse) + ACI
#                intervals (MAPIE).   Objective: validation MSE.

SHARED: lags, learning_rate, dropout (see COMMON PARAMETERS)
LSTM: lstm_depth, lstm_width (see COMMON PARAMETERS)
(no additional searched parameters beyond the shared/MDN grid)

FIXED (NOT searched)
    weighting=none, batch_size=512, point_head_layers=[32]
    max_epochs=1000, max_norm=30.0, patience=30, patience_scheduler=10,
    factor_scheduler=0.5

CLI-CONTROLLED (not searched)
    --horizon    forecast horizon h (default 1). Part of the artifact path
                 (outputs/simulations/<process>/horizon_<h>/...).
    --density    MDN mixture component density: gaussian | student | skewt | skewnorm
                 (default skewt).
    --weighting  tail-weighting scheme for fit(): both | sampler | loss | none
                 (default both)
                   both    = WeightedRandomSampler + weighted NLL (original scheme)
                   sampler = sampler only (NLL unweighted)
                   loss    = weighted NLL only (plain shuffle)
                   none    = no weighting at all
                 Non-'both' weightings nest under a <weighting> subfolder so they
                 self-separate; 'both' keeps its files at the <density> level.
    --process    DGP tag -> outputs/simulations/<process>/horizon_<h>/
                 (default mar01).
                 Selects the DGP and (re)builds the series.
    --n_trials   max Optuna trials for the active study (required unless --evaluate).
    --evaluate   skip tuning; reload the saved config and run the test-set
                 evaluation directly. Honored by every mode below.

    --model      which model to tune/evaluate (default 'mdn'); each non-'mdn' choice
                 REPLACES the MDN density flow:
      mdn              the MDN density backbone (uses --density/--weighting).
      point_head       tune + evaluate the regression point head on the FROZEN
                       density backbone (see POINT HEAD block). Only supported with
                       --density skewt; --density/--weighting are
                       optional and default to skewt/both (an explicit different
                       density is an error). If the density config does not exist
                       yet, the base model is tuned first. With --evaluate,
                       reloads point_head_config.yaml (incl. its density params)
                       and skips tuning.
      flexzboost       tune + evaluate the FlexCode(XGBoost) baseline (see FLEXZBOOST
                       block). -> <process>/flexzboost/
      kcde             tune + evaluate the kernel-CDE baseline (see KCDE block).
                       -> <process>/kcde/
      flow             tune + evaluate the conditional-NSF baseline (see FLOW block).
                       -> <process>/flow/
      conformal        tune + evaluate the unweighted point predictor + ACI intervals
                       (see CONFORMAL block). -> <process>/conformal/ (per-gamma eval
                       JSONs under conformal/gamma_<tag>/)
    (the baseline models are independent of --density / --weighting)
======================================================================
"""

import argparse
import faulthandler
import os
from pathlib import Path

faulthandler.enable()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
import yaml

from src.forecast_methods.Conformal import (
    CONFORMAL_ACI_GAMMAS,
    CONFORMAL_ALPHA,
    conformal_regions,
    fit_mse_unweighted,
)
from src.forecast_methods.FlexZBoost import (
    FLEXZBOOST_BUMP_GRID,
    FLEXZBOOST_LR_GRID,
    FLEXZBOOST_MAX_BASIS,
    FLEXZBOOST_SHARPEN_GRID,
    build_and_fit_flexzboost,
    flexzboost_grid_bounds,
    recalibrate_test_flexzboost,
)
from src.forecast_methods.Flow import (
    FLOW_BINS_GRID,
    FLOW_DEPTH_RANGE,
    FLOW_LR_GRID,
    FLOW_TRANSFORMS_RANGE,
    FLOW_WIDTH_GRID,
    build_flow,
    fit_flow_scalers,
    recalibrate_test_flow,
    train_flow,
)
from src.forecast_methods.KCDE import (
    KCDE_BANDWIDTH_GRID,
    KCDE_KERNELS,
    kcde_grid,
    kcde_val_loss,
    recalibrate_test_kcde,
)
from src.forecast_methods.MDN import SATURATED_FAMILIES as _SATURATED_FAMILIES
from src.forecast_methods.MDN import build_lstm as _build_lstm_model
from src.forecast_methods.MDN import fit_and_score
from src.forecast_methods.MDN import recalibrate_test as _recalibrate_test_mdn
from src.forecast_methods.utils import (
    SPLIT_MAX_LAGS,
    fit_and_apply_recalibrator,
    prepare_tensors,
)
from src.markov_switching_bubble.markov_switching_bubble import MarkovSwitchingBubble
from src.metrics.density import (
    hellinger_regions as _hellinger_regions,
)
from src.metrics.density import ise_regions as _ise_regions
from src.metrics.density import kl_regions as _kl_regions
from src.metrics.density import point_estimates as _point_estimates
from src.metrics.point import mae_regions as _mae_regions
from src.metrics.point import mse_regions as _mse_regions
from src.metrics.point import no_change_mae as _no_change_mae
from src.metrics.point import no_change_mse as _no_change_mse
from src.metrics.point import relative_mae as _relative_mae
from src.metrics.point import relative_mse as _relative_mse
from src.metrics.quantile import PINBALL_LEVELS
from src.metrics.quantile import cdf_from_density as _cdf_from_density
from src.metrics.quantile import (
    equal_tailed_regions as _equal_tailed_regions,
)
from src.metrics.quantile import pinball_samples as _pinball_samples
from src.metrics.quantile import (
    quantiles_from_cdf as _quantiles_from_cdf,
)
from src.metrics.quantile import (
    sample_total_mean as _sample_total_mean,
)
from src.metrics.quantile import shortest_regions as _shortest_regions
from src.metrics.regions import coverage_regions as _coverage_regions
from src.metrics.regions import hdr_regions as _hdr_regions
from src.metrics.regions import interval_metrics as _conformal_interval_metrics
from src.metrics.regions import length_regions as _length_regions
from src.metrics.regions import winkler_regions as _winkler_regions
from src.optuna.runner import run_optuna_study
from src.optuna.search_space import (
    suggest_backbone_arch,
    suggest_dropout,
    suggest_lags,
    suggest_learning_rate,
)
from src.optuna.study_utils import resolved_params as _resolved_params
from src.results.io import (
    calibration_frame as _calibration_frame,
)
from src.results.io import (
    density_predictions_frame as _density_predictions_frame_impl,
)
from src.results.io import format_number_4_digits
from src.results.io import (
    load_model_weights as _load_model_weights,
)
from src.results.io import (
    point_predictions_frame as _point_predictions_frame,
)
from src.results.io import (
    save_density_parquets as _save_density_parquets_impl,
)
from src.results.io import (
    save_model_weights as _save_model_weights,
)
from src.results.io import (
    save_point_parquet as _save_point_parquet,
)
from src.results.io import to_native as _to_native
from src.results.io import write_json as _write_json
from src.results.io import write_yaml as _write_yaml
from src.results.paths import ArtifactPaths
from src.results.paths import algo_dir as _algo_dir_impl
from src.results.paths import density_weights_path as _density_weights_path
from src.results.paths import pit_calibration_path as _pit_calibration_path
from src.results.paths import point_head_weights_path as _point_head_weights_path
from src.results.paths import point_predictions_path as _point_predictions_path
from src.results.paths import raw_densities_path as _raw_densities_path
from src.results.paths import test_densities_path as _test_densities_path
from src.results.reporting import aggregate_over_seeds as _aggregate_over_seeds
from src.results.reporting import fmt_cell as _fmt_cell
from src.results.reporting import print_eval_tables as _print_eval_tables
from src.results.reporting import (
    write_multiseed_eval_jsons as _write_multiseed_eval_jsons,
)
from src.stable_mar.stable_mar import stablemar as sm
from src.theoretical.cauchy_closed_form import cauchy_ar1_predictive_density
from src.theoretical.theoretical_quantiles import compute_theoretical_quantiles
from src.utils.setup_config_device import (
    get_allowed_cpu_count,
    set_seed,
    setup_config_device,
    setup_device,
)
from src.utils.setup_logger import setup_logger

logger = setup_logger()

# Setup
device = setup_device()
cpu_count = get_allowed_cpu_count()
n_process = setup_config_device(cpu_count)

# Config
_SIMULATIONS_ROOT = Path("outputs/simulations")
_CONFIGS_ROOT = Path("configs")

with open(_CONFIGS_ROOT / "run_config.yaml") as _f:
    _ROOT_CFG = yaml.safe_load(_f)

# DGP definitions
with open(_CONFIGS_ROOT / "dgp_config.yaml") as _f:
    _DGP_CFG = yaml.safe_load(_f)

SEED = _ROOT_CFG["seed"]
set_seed(SEED)

# Seeds used for the multi-seed test evaluation.
EVAL_SEEDS = list(_ROOT_CFG.get("eval_seeds", [SEED]))

PROPORTIONS = tuple(_ROOT_CFG["proportions"])
N_GRID = _ROOT_CFG["n_grid"]

# Grid size for the persisted per-seed density parquets. Equal to N_GRID so the
# saved densities are exactly what was scored and --evaluate reproduces a full
# run's numbers; lower it to trade parquet size for that fidelity.
PRED_GRID_SIZE = N_GRID

_train_cfg = _ROOT_CFG["training"]
BATCH_SIZE = _train_cfg["batch_size"]
MAX_EPOCHS = _train_cfg["max_epochs"]
MAX_NORM = _train_cfg["max_norm"]
PATIENCE = _train_cfg["patience"]
PATIENCE_SCHEDULER = _train_cfg["patience_scheduler"]
FACTOR_SCHEDULER = _train_cfg["factor_scheduler"]

POINT_HEAD_LAYERS = _ROOT_CFG["model"]["point_head_layers"]

# Data: simulated series, selected by --process
ALPHA, BETA, SIGMA, N = None, None, None, None

# Process registry
PROCESSES = {}
for _name, _p in _DGP_CFG.get("processes", {}).items():
    _kind = _p.get("kind", "stable_mar")
    if _kind == "stable_mar":
        PROCESSES[_name] = dict(
            kind=_kind,
            order=tuple(_p["order"]),
            causal=list(_p["causal"]),
            noncausal=list(_p["noncausal"]),
            closed_form=_p["closed_form"],
        )
    elif _kind == "ms_bubble":
        PROCESSES[_name] = dict(
            kind=_kind,
            closed_form=_p["closed_form"],
            params=dict(_p["params"]),
        )
    else:
        raise SystemExit(
            f"Unknown DGP kind {_kind!r} for process {_name!r} in dgp_config.yaml."
        )

DGP = None
SERIES = None


def _resolve_dgp(process):
    """Map a --process tag to a registry entry.

    Exact match first; otherwise the longest registry key that the tag starts
    with (so folder-only variants like 'mar01_h2' still resolve to their DGP).
    """
    if process in PROCESSES:
        return process, PROCESSES[process]
    for key in sorted(PROCESSES, key=len, reverse=True):
        if process.startswith(key):
            return key, PROCESSES[key]
    raise SystemExit(
        f"Unknown --process {process!r}: expected one of {sorted(PROCESSES)} "
        f"(or a tag prefixed by one, e.g. 'mar01_h2')."
    )


def build_series(process, seed=None):
    """Select the DGP for `process`, simulate the series, and cache both globally.

    Sets the module globals DGP (the resolved registry entry, plus its name) and
    SERIES (the 1-D float32 trajectory). Re-seeds first so the series is
    reproducible. `seed` defaults to the global SEED (matching the original mar01
    setup); pass a different seed to draw a fresh realization of the same DGP,
    which is how the multi-seed evaluation obtains process-level variance.
    """
    global DGP, SERIES, ALPHA, BETA, SIGMA, N
    name, spec = _resolve_dgp(process)

    _s = _DGP_CFG["series"]
    N, ALPHA, BETA, SIGMA = _s["n"], _s["alpha"], _s["beta"], _s["sigma"]

    seed = SEED if seed is None else seed
    set_seed(seed)  # deterministic innovations for this seed

    if spec["kind"] == "ms_bubble":
        model = MarkovSwitchingBubble(**spec["params"])
        SERIES = model.simulate(N, seed=seed).astype(np.float32)
        DGP = {"name": name, **spec, "model": model}
        logger.info(
            f"DGP {name}: kind=ms_bubble | params={spec['params']} | "
            f"closed_form={spec['closed_form']} | tail_index={model.tail_index:.2f} | "
            f"series N={len(SERIES)} | seed={seed}"
        )
        return SERIES

    par = list(spec["causal"]) + list(spec["noncausal"]) + [ALPHA, BETA, SIGMA]

    mar = sm(spec["order"])
    mar.model = "MAR"
    mar.par = par
    traj = mar.generate(N).trajectory.dropna().reset_index(drop=True)
    SERIES = traj.to_numpy(dtype=np.float32).ravel()
    DGP = {"name": name, **spec}

    logger.info(
        f"DGP {name}: order={spec['order']} | causal={spec['causal']} | "
        f"noncausal={spec['noncausal']} | closed_form={spec['closed_form']} | "
        f"par={par} | series N={len(SERIES)} | seed={seed}"
    )
    return SERIES


build_series("mar01")

# CLI-driven settings: no defaults.
HORIZON = None
DENSITY = None
WEIGHTING = None

APPLICATION_PINBALL_LEVELS = None

# Per-lag tensor cache:
_DATA_CACHE = {}


def get_data(lags):
    """Build (and cache) the train/cali/test tensors for a given number of lags."""
    if lags not in _DATA_CACHE:
        (
            X_train,
            y_train,
            X_cali,
            y_cali,
            X_test,
            y_test,
            weights_train,
            weights_val,
        ) = prepare_tensors(
            df=None,
            X=SERIES,
            y=SERIES,
            lags=lags,
            horizon=HORIZON,
            proportions=PROPORTIONS,
            device=device,
        )
        _DATA_CACHE[lags] = dict(
            X_train=X_train,
            y_train=y_train,
            X_cali=X_cali,
            y_cali=y_cali,
            X_test=X_test,
            y_test=y_test,
            weights_train=weights_train,
            weights_val=weights_val,
        )
        logger.info(
            f"Built tensors for lags={lags} | train={len(X_train)} "
            f"cali/val={len(X_cali)} test={len(X_test)}"
        )
    return _DATA_CACHE[lags]


# Process-scoped output root.
PROCESS = None
OUT_DIR = None

REPLAY_CACHED = False
RESCORE_ONLY = False
FORCE_REFIT = False
SAVE_RAW = True


def init(horizon, process, density, weighting, name=None):
    """Apply the CLI settings, select the DGP and create the output root.

    ``name`` flattens the output root to outputs/simulations/<name>/.
    """
    global HORIZON, PROCESS, OUT_DIR, DENSITY, WEIGHTING
    HORIZON = horizon
    PROCESS = process
    DENSITY = density
    WEIGHTING = weighting
    OUT_DIR = (
        _SIMULATIONS_ROOT / name
        if name
        else _SIMULATIONS_ROOT / PROCESS / f"horizon_{HORIZON}"
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _DATA_CACHE.clear()  # horizon/process change the tensors, so invalidate any cache
    build_series(process)  # selects the DGP and (re)builds SERIES
    logger.info(
        f"PROCESS={PROCESS} | HORIZON={HORIZON} | DENSITY={DENSITY} | "
        f"WEIGHTING={WEIGHTING} | DGP={DGP['name']} | output root: {OUT_DIR}/"
    )


def backbone_dir(backbone):
    """Per-run artifact directory.

    outputs/simulations/<PROCESS>/horizon_<h>/mdn/            (Skew-t, both)
    outputs/simulations/<PROCESS>/horizon_<h>/ablation/<w>/   (loss ablation)
    outputs/simulations/<PROCESS>/horizon_<h>/ablation/<DENSITY>/   (ablation densities)

    The Skew-t density is the headline model and lives at the top level as
    outputs/simulations/<PROCESS>/horizon_<h>/mdn/ (a sibling of the
    flexzboost/kcde/flow/conformal baselines). The ablation densities
    (gaussian / skewnorm / student) live under
    outputs/simulations/<PROCESS>/horizon_<h>/ablation/<DENSITY>/.

    weighting=both keeps its files at that level; non-default weightings nest in
    their own subfolder, so weighting variants under the same --process don't
    clobber.
    """
    if DENSITY == "skewt":
        if WEIGHTING == "both":
            d = OUT_DIR / "mdn"
        else:
            d = OUT_DIR / "ablation" / WEIGHTING
    else:
        d = OUT_DIR / "ablation" / DENSITY
        if WEIGHTING != "both":
            d = d / WEIGHTING
    d.mkdir(parents=True, exist_ok=True)
    return d


def _artifact_key(backbone):
    """Trials-tracker key mirroring the artifact path."""
    if DENSITY == "skewt":
        if WEIGHTING == "both":
            return "mdn"
        return f"ablation/{WEIGHTING}"
    base = f"ablation/{DENSITY}"
    if WEIGHTING != "both":
        base = f"{base}/{WEIGHTING}"
    return base


_backbone_paths = ArtifactPaths(backbone_dir)
_study_db_url = _backbone_paths.study_db_url


def _algo_dir(name, subdir=None):
    """Baseline artifact directory <PROCESS>/<name>[/<subdir>]."""
    return _algo_dir_impl(OUT_DIR, name, subdir)


# Objective helpers
def _fit_and_score(mdn, data, lr, trial):
    """Train one model and return the validation NLL (objective value)."""
    return fit_and_score(
        mdn,
        data,
        lr,
        trial,
        max_epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        max_norm=MAX_NORM,
        patience=PATIENCE,
        patience_scheduler=PATIENCE_SCHEDULER,
        factor_scheduler=FACTOR_SCHEDULER,
        weighting=WEIGHTING,
        logger=logger,
    )


def _suggest_shared(trial):
    """SHARED hyperparameters searched for all backbones."""
    return {
        "lags": suggest_lags(trial),
        "n_mixtures": trial.suggest_int("n_mixtures", 1, 10),
        "learning_rate": suggest_learning_rate(trial),
        "dropout": suggest_dropout(trial),
    }


# Model builders
def _build_lstm(p, n_jobs):
    return _build_lstm_model(p, device, n_jobs, DENSITY, POINT_HEAD_LAYERS)


BUILDERS = {
    "lstm": _build_lstm,
}


def objective_lstm(trial):
    """Optuna objective for the LSTM-backbone MDN."""
    set_seed(SEED)
    s = _suggest_shared(trial)
    data = get_data(s["lags"])

    s = suggest_backbone_arch(trial, "lstm", s)

    mdn = _build_lstm(s, n_jobs=1)
    return _fit_and_score(mdn, data, s["learning_rate"], trial)


OBJECTIVES = {
    "lstm": objective_lstm,
}

_config_path = _backbone_paths.config_path

_TRAINED_DENSITY = {}
_TRAINED_POINT_HEAD = {}


def _params_key(params):
    return tuple(sorted((k, str(_to_native(v))) for k, v in params.items()))


def save_config_yaml(backbone, params, val_nll=None, n_params=None):
    """Persist the optimal config to <backbone>_config.yaml."""
    cfg = {
        "backbone": backbone,
        "params": {k: _to_native(v) for k, v in params.items()},
        "fixed": {
            "horizon": HORIZON,
            "density": DENSITY,
            "weighting": WEIGHTING,
            "batch_size": BATCH_SIZE,
            "point_head_layers": POINT_HEAD_LAYERS,
            "max_epochs": MAX_EPOCHS,
            "max_norm": MAX_NORM,
            "patience": PATIENCE,
            "patience_scheduler": PATIENCE_SCHEDULER,
            "factor_scheduler": FACTOR_SCHEDULER,
        },
        "val": {
            "nll": _to_native(val_nll),
            "n_density_params": _to_native(n_params),
        },
    }
    path = _write_yaml(_config_path(backbone), cfg)
    logger.info(f"[{backbone}] optimal config saved to {path}")
    return path


def load_config_yaml(backbone):
    """Load <backbone>_config.yaml and return the params dict the builders expect."""
    global DENSITY, WEIGHTING
    path = _config_path(backbone)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    params = dict(cfg["params"])
    params.setdefault("lags", cfg.get("fixed", {}).get("lags"))
    DENSITY = cfg.get("fixed", {}).get("density", DENSITY)
    WEIGHTING = cfg.get("fixed", {}).get("weighting", WEIGHTING)
    logger.info(
        f"[{backbone}] config loaded from {path} "
        f"(density={DENSITY}, weighting={WEIGHTING})"
    )
    return params


# Weighting schemes that are ablation cells rather than the headline ('both').
_WEIGHTING_ABLATION = ("none", "sampler", "loss")


def _is_saturated_ablation():
    """Report whether the density is a saturated skew-t ablation family.

    Gaussian / skewnorm / student reuse the skew-t config and are retrained, never
    tuned.
    """
    return DENSITY in _SATURATED_FAMILIES


def _is_weighting_ablation():
    """Report whether the run is a tail-weighting ablation cell.

    A plain skew-t under a non-"both" scheme (none / sampler / loss); like the
    density families these reuse the headline config and only retrain.
    """
    return DENSITY == "skewt" and WEIGHTING in _WEIGHTING_ABLATION


def _reuses_flagship_config():
    """Report whether the ablation reuses the headline skew-t/both config.

    True for a density family or a weighting scheme, i.e. never tuned.
    """
    return _is_saturated_ablation() or _is_weighting_ablation()


def _ablation_label():
    """Short description of the active reuse-the-flagship ablation, for logs."""
    if _is_saturated_ablation():
        return f"density ablation '{DENSITY}' (saturated skew-t)"
    if _is_weighting_ablation():
        return f"weighting ablation '{WEIGHTING}' (skew-t)"
    return "ablation"


def _skewt_both_config_path():
    """Path to the headline skew-t / weighting=both config.

    Independent of the current DENSITY / WEIGHTING globals, which the ablations
    reuse. Mirrors backbone_dir()'s skew-t/both branch.
    """
    return OUT_DIR / "mdn" / "config.yaml"


def load_skewt_both_config(backbone):
    """Return the params of the headline skew-t (weighting=both) config.

    Does not mutate the DENSITY / WEIGHTING globals: the ablations keep their own
    label but train the skew-t architecture.
    """
    path = _skewt_both_config_path()
    if not path.exists():
        raise SystemExit(
            f"[{backbone}] {_ablation_label()} needs the headline skew-t config at "
            f"{path}, but it does not exist. Run the skew-t model first "
            f"(--density skewt --weighting both) for this process/horizon."
        )
    with open(path) as f:
        cfg = yaml.safe_load(f)
    params = dict(cfg["params"])
    params.setdefault("lags", cfg.get("fixed", {}).get("lags"))
    logger.info(
        f"[{backbone}] {_ablation_label()} reuses skew-t/both config from {path} "
        f"(retrain only, no tuning)"
    )
    return params


def _config_block(backbone, params):
    """Format the best params as a paste-ready Python config block."""
    s = params
    lines = [
        f"# === best {backbone} config (val NLL minimized) ===",
        f"LAGS = {s['lags']}",
        f"N_MIXTURES = {s['n_mixtures']}",
        f"DROPOUT = {format_number_4_digits(s['dropout'])}",
        "FIT_KWARGS = dict(",
        f"    max_epochs={MAX_EPOCHS},",
        f"    learning_rate={s['learning_rate']:.3e},",
        f"    batch_size={BATCH_SIZE},",
        f"    max_norm={MAX_NORM},",
        f"    patience={PATIENCE},",
        f"    patience_scheduler={PATIENCE_SCHEDULER},",
        f"    factor_scheduler={FACTOR_SCHEDULER},",
        ")",
    ]
    hidden = [s["lstm_width"]] * s["lstm_depth"]
    lines.append(f"HIDDEN_LAYERS = {hidden}  # len = n_layers, entry = hidden_size")
    return "\n".join(lines)


def run_study(backbone, n_trials, minutes_per_backbone):
    # Always persist the study in <backbone>/<density>/study.db so runs are
    """Run the Optuna study for the active MDN backbone."""
    study = run_optuna_study(
        study_name=f"mdn_{backbone}_{DENSITY}",
        label=f"{backbone} ({DENSITY})",
        progress_tag=f"[{backbone}]",
        out_dir=backbone_dir(backbone),
        storage=_study_db_url(backbone),
        tracker_key=_artifact_key(backbone),
        objective=OBJECTIVES[backbone],
        n_trials=n_trials,
        timeout_minutes=minutes_per_backbone,
        seed=SEED,
    )

    best = study.best_trial
    resolved = _resolved_params(study)
    logger.info(f"\n[{backbone}] best val NLL: {best.value:.5f}")
    logger.info(
        f"[{backbone}] density params: {best.user_attrs.get('n_density_params', 'NA')}"
    )
    block = _config_block(backbone, resolved)
    logger.info(f"\n{block}\n")

    # Persist the optimal config.
    save_config_yaml(
        backbone,
        resolved,
        val_nll=best.value,
        n_params=best.user_attrs.get("n_density_params"),
    )

    return study


def get_density_model(backbone, params):
    """Return the tuned density model at SEED, trained on first use and cached."""
    key = (backbone, DENSITY, WEIGHTING, _params_key(params))
    if key in _TRAINED_DENSITY:
        return _TRAINED_DENSITY[key]

    w = _density_weights_path(backbone_dir(backbone))
    if not FORCE_REFIT and w.exists():
        mdn = BUILDERS[backbone](params, n_jobs=n_process)
        if _load_model_weights(w, mdn, params):
            logger.info(f"[{backbone}] weights restored from {w} -- no training")
            mdn.to(device)
            _TRAINED_DENSITY[key] = mdn
            return mdn

    logger.info(f"[{backbone}] training the best config...")
    set_seed(SEED)
    data = get_data(params["lags"])
    mdn = BUILDERS[backbone](params, n_jobs=n_process)
    mdn.fit(
        X_train=data["X_train"],
        y_train=data["y_train"],
        X_val=data["X_cali"],
        y_val=data["y_cali"],
        weights_train=data["weights_train"],
        weights_val=data["weights_val"],
        max_epochs=MAX_EPOCHS,
        learning_rate=params["learning_rate"],
        batch_size=BATCH_SIZE,
        max_norm=MAX_NORM,
        patience=PATIENCE,
        patience_scheduler=PATIENCE_SCHEDULER,
        factor_scheduler=FACTOR_SCHEDULER,
        weighting=WEIGHTING,
    )
    _save_model_weights(w, mdn, params)
    _TRAINED_DENSITY[key] = mdn
    return mdn


# Test-set evaluation: recalibrate on the cali set, predict on the test set.
def _marginal_quantiles(lower, upper):
    """Return the DGP's theoretical marginal quantiles at (lower, upper).

    Closed form for the stable MAR/MARMA processes, long-simulation estimate for the
    Markov-switching bubble. Realization-independent by construction.
    """
    if DGP["kind"] == "ms_bubble":
        lo, hi = DGP["model"].marginal_quantiles([lower, upper])
        return float(lo), float(hi)

    theo_q = compute_theoretical_quantiles(
        phi_vec=DGP["causal"],
        psi_vec=DGP["noncausal"],
        alpha=ALPHA,
        beta=BETA,
        sigma=SIGMA,
    )
    return float(theo_q["Lower Quantiles"][lower]), float(
        theo_q["Upper Quantiles"][upper]
    )


def _theoretical_band():
    """Theoretical marginal [10%, 90%] band.

    Matches the brent application, which uses the q10/q90 of its train+calibration
    segment, so the 'Tails' column covers a comparable share of the test set in
    both the simulation and the application tables (~20%).
    """
    q10, q90 = _marginal_quantiles(0.10, 0.90)
    logger.info(
        f"Theoretical marginal band [10%, 90%] = "
        f"[{format_number_4_digits(q10)}, {format_number_4_digits(q90)}]"
    )
    return q10, q90


def _build_eval_grid(data, q05, q95):
    """Build the evaluation grid for the current ``data`` split."""
    if DGP.get("kind") == "real":
        s = np.asarray(SERIES, dtype=np.float64)
        lo, hi = float(s.min()), float(s.max())
        pad = 0.2 * (hi - lo)
        lo, hi = lo - pad, hi + pad
        if s.min() > 0:
            lo = max(lo, 0.0)
    else:
        lo, hi = _marginal_quantiles(0.001, 0.999)
    return np.linspace(lo, hi, N_GRID).astype(np.float32)


def _density_predictions_frame(seed, y_true, x_last_lag, grid_y, density, x_test=None):
    """Return one seed's uncorrected test densities in long format.

    Subsampled onto PRED_GRID_SIZE points.
    """
    return _density_predictions_frame_impl(
        seed, y_true, x_last_lag, grid_y, density, PRED_GRID_SIZE, x_test=x_test
    )


def _density_frames(seed, data, grid_y, recalibrated_density, artifacts):
    """Return the (raw, recalibrated, calibration) frames for one fit.

    The raw frame is skipped under --no_save_raw: it is the same size as the
    recalibrated one and only the calibrator-refit path reads it.
    """
    y_test = data["y_test"].detach().cpu().numpy()
    x_test = np.asarray(artifacts["X_test"], dtype=np.float32)
    x_last_lag = x_test[:, -1]
    return (
        _density_predictions_frame(
            seed, y_test, x_last_lag, grid_y, artifacts["raw"], x_test=x_test
        )
        if SAVE_RAW
        else None,
        _density_predictions_frame(
            seed, y_test, x_last_lag, grid_y, recalibrated_density
        ),
        _calibration_frame(seed, artifacts["pit_cali"], artifacts["X_cali"]),
    )


def _save_density_parquets(out_dir, raw_frames, test_frames, cali_frames):
    """Write raw_densities / test_densities / pit_calibration for one run."""
    return _save_density_parquets_impl(
        out_dir,
        raw_frames,
        test_frames,
        cali_frames,
    )


def _recalibrate_test(mdn, data, grid_y, artifacts=None):
    """Recalibrate on the calibration split; return (raw, recalibrated)."""
    return _recalibrate_test_mdn(
        mdn, data, grid_y, device, n_process, artifacts=artifacts
    )


def _has_density_cache(out_dir):
    """Report whether raw densities and PITs are on disk to replay."""
    return (
        _raw_densities_path(out_dir).exists()
        and _pit_calibration_path(out_dir).exists()
    )


def _replay_from_cache(out_dir, q05, q95):
    """Re-score a finished run from its parquets, without refitting any model.

    Refits the recalibrator on the cached calibration PITs and re-applies it to
    the cached raw densities, then recomputes the full metric suite. This is what
    --evaluate does when the parquets are on disk: a change to the calibrator can
    be re-scored in seconds instead of retraining every seed.

    Returns (seeds, per_seed_metrics, raw_frames, test_frames, cali_frames).
    """
    raw = pd.read_parquet(_raw_densities_path(out_dir))
    cali = pd.read_parquet(_pit_calibration_path(out_dir))
    seeds = sorted(raw["seed"].unique().tolist())
    logger.info(
        f"replaying {_raw_densities_path(out_dir)} -- refitting the calibrator on "
        f"{len(cali)} cached PITs over seeds {seeds} (no model refit)"
    )

    per_seed, raw_frames, test_frames, cali_frames = [], [], [], []
    for s in seeds:
        g = raw[raw["seed"] == s]
        c = cali[cali["seed"] == s]
        grid_y = np.asarray(g["grid_y"].iloc[0], dtype=np.float32)
        x_test = np.stack(g["x_test"].to_numpy()).astype(np.float32)
        x_cali = np.stack(c["x_cali"].to_numpy()).astype(np.float32)
        raw_density = np.stack(g["density"].to_numpy()).astype(np.float64)

        recalibrated_density = fit_and_apply_recalibrator(
            x_cali, c["pit"].to_numpy(), x_test, raw_density, grid_y, n_process
        )

        # The closed-form true density conditions on the series, so redraw this
        # seed's realization (cheap -- no training) before scoring.
        if DGP["closed_form"]:
            build_series(PROCESS, seed=s)
            _DATA_CACHE.clear()
        data = {
            "y_test": torch.as_tensor(g["y_true"].to_numpy(), dtype=torch.float32),
            "X_test": torch.as_tensor(x_test),
        }
        per_seed.append(
            _density_test_metrics(recalibrated_density, data, grid_y, q05, q95)
        )

        raw_frames.append(g)
        test_frames.append(
            _density_predictions_frame(
                s,
                g["y_true"].to_numpy(),
                g["x_last_lag"].to_numpy(),
                grid_y,
                recalibrated_density,
            )
        )
        cali_frames.append(c)

    return seeds, per_seed, raw_frames, test_frames, cali_frames


def _has_test_density_cache(out_dir):
    """Report whether recalibrated test densities are on disk."""
    return _test_densities_path(out_dir).exists()


def _log_replay_fallback(out_dir):
    """Say out loud that --evaluate is about to retrain rather than replay."""
    logger.warning(
        f"--evaluate: no {_raw_densities_path(out_dir)} to replay -- retraining "
        f"every seed from the saved config. Raw densities are saved by default, "
        f"so this run predates them or used --no_save_raw; --rescore re-scores "
        f"the saved calibrated densities without any fitting."
    )


def _require_test_density_cache(out_dir):
    """Under --rescore, refuse to silently fall back to fitting anything."""
    if not _has_test_density_cache(out_dir):
        logger.warning(
            f"--rescore: no {_test_densities_path(out_dir)} -- nothing to "
            f"re-score, skipping (run --evaluate --refit first to produce it)."
        )
        raise SystemExit(0)


def _rescore_from_cache(out_dir, q05, q95):
    """Re-score a finished run straight from its calibrated test densities.

    Returns (seeds, per_seed_metrics).
    """
    test = pd.read_parquet(_test_densities_path(out_dir))
    seeds = sorted(test["seed"].unique().tolist())
    logger.info(
        f"rescoring {_test_densities_path(out_dir)} over seeds {seeds} "
        f"(no model refit, no recalibrator refit)"
    )

    per_seed = []
    for s in seeds:
        g = test[test["seed"] == s]
        grid_y = np.asarray(g["grid_y"].iloc[0], dtype=np.float32)
        density = np.stack(g["density"].to_numpy()).astype(np.float64)

        # The closed-form true density conditions on the series, so redraw this
        # seed's realization (cheap -- no training) before scoring.
        if DGP["closed_form"]:
            build_series(PROCESS, seed=s)
            _DATA_CACHE.clear()

        # Only the last lag is persisted, and it is all the metrics need: the
        # sample split, the no-change benchmark and the closed-form conditioning
        # all read X_test[:, -1].
        x_last = g["x_last_lag"].to_numpy().astype(np.float32)[:, None]
        data = {
            "y_test": torch.as_tensor(g["y_true"].to_numpy(), dtype=torch.float32),
            "X_test": torch.as_tensor(x_last),
        }
        per_seed.append(_density_test_metrics(density, data, grid_y, q05, q95))

    return seeds, per_seed


def _point_head_metric_keys(m):
    """Rename the shared point metrics to the point_head_eval.json key names."""
    return {
        "mse_point_head_rel": m["mse_rel"],
        "mae_point_head_rel": m["mae_rel"],
    }


def _log_point_head_seed(backbone, seed, m):
    logger.info(
        f"[{backbone}] seed {seed} | point-head MSE(all)="
        f"{format_number_4_digits(m['mse_point_head']['all'])} | rel(all)="
        f"{format_number_4_digits(m['mse_point_head_rel']['all'])} | MAE(all)="
        f"{format_number_4_digits(m['mae_point_head']['all'])} | rel_mae(all)="
        f"{format_number_4_digits(m['mae_point_head_rel']['all'])}"
    )


def _np_last_lag(data):
    """Return the last observed lag column of X_test as float64."""
    return data["X_test"].detach().cpu().numpy()[:, -1].astype(np.float64)


def _has_point_cache(out_dir):
    """Report whether point forecasts are on disk to re-score."""
    return _point_predictions_path(out_dir).exists()


def _require_point_cache(out_dir):
    """Under --rescore, refuse to fall back to refitting a point predictor."""
    if not _has_point_cache(out_dir):
        logger.warning(
            f"--rescore: no {_point_predictions_path(out_dir)} -- nothing to "
            f"re-score, skipping (run --evaluate --refit first to produce it)."
        )
        raise SystemExit(0)


def _point_region_metrics(pred, y_true, data, q05, q95):
    """MSE/MAE by bulk/tail region plus the no-change benchmark and relatives."""
    grid_y = _build_eval_grid(data, q05, q95)
    y_full = np.asarray(y_true, dtype=np.float64)
    keep = (y_full >= float(grid_y[0])) & (y_full <= float(grid_y[-1]))
    n_dropped = int((~keep).sum())
    if n_dropped:
        logger.info(
            f"dropping {n_dropped}/{keep.size} test targets outside the grid "
            f"({100 * n_dropped / max(keep.size, 1):.2f}%)"
        )
        keep_t = torch.as_tensor(keep)
        # Only the test tensors are indexed by `keep`: the fitted path passes the
        # whole split (train/cali included), the rescore path passes X_test alone.
        data = {k: data[k][keep_t] for k in ("X_test", "y_test") if k in data}
        y_true = y_full[keep]
        pred = np.asarray(pred, dtype=np.float64)[keep]

    mask = _sample_region_mask(data, q05, q95)
    mse = _mse_regions(pred, y_true, mask)
    mae = _mae_regions(pred, y_true, mask)
    mse_nc = _no_change_mse(data, y_true, mask)
    mae_nc = _no_change_mae(data, y_true, mask)
    return {
        "mse_rel": _relative_mse(mse, mse_nc),
        "mae_rel": _relative_mae(mae, mae_nc),
    }


def _rescore_point_from_cache(out_dir, q05, q95):
    """Re-score a finished point-forecast run from point_predictions.parquet.

    The point-forecast twin of _rescore_from_cache: the stored predictions are
    final (no calibration step exists for a point forecast), so the metric suite
    is recomputed exactly as it was at write time, with no refit of any kind.

    Returns (seeds, per_seed_metrics).
    """
    df = pd.read_parquet(_point_predictions_path(out_dir))
    seeds = sorted(df["seed"].unique().tolist())
    logger.info(
        f"rescoring {_point_predictions_path(out_dir)} over seeds {seeds} "
        f"(no model refit)"
    )

    per_seed = []
    for s in seeds:
        g = df[df["seed"] == s]
        y = g["y_true"].to_numpy(dtype=np.float64)
        pred = g["pred"].to_numpy(dtype=np.float64)
        x_last = g["x_last_lag"].to_numpy().astype(np.float32)[:, None]
        data = {"X_test": torch.as_tensor(x_last)}
        per_seed.append(_point_region_metrics(pred, y, data, q05, q95))

    return seeds, per_seed


def _test_conditioning_indices(data):
    """Return the time index of the last observed value for each test sample.

    Lets the exact ms_bubble predictive condition on SERIES[0..tau] with target
    SERIES[tau + HORIZON]. Reconstructs prepare_tensors' split arithmetic.
    """
    n_ref = len(SERIES) - HORIZON + 1
    gap = SPLIT_MAX_LAGS + HORIZON - 1
    t_train_end = int(PROPORTIONS[0] * n_ref)
    t_val_end = t_train_end + int(PROPORTIONS[1] * n_ref)
    t_test = np.arange(t_val_end + gap, n_ref)

    n_test = len(data["X_test"])
    assert n_test == len(t_test), (
        f"test-split reconstruction mismatch: expected {len(t_test)} "
        f"test samples, data has {n_test}"
    )
    cond_idx = t_test - 1
    x_last = data["X_test"].detach().cpu().numpy()[:, -1]
    assert np.array_equal(SERIES[cond_idx], x_last), (
        "test-sample time indices do not match X_test's last lag"
    )
    return cond_idx


def _pinball_levels():
    """Quantile-score levels for the active series.

    Applications can explicitly select reportable levels; simulations use the
    full two-sided set.
    """
    if APPLICATION_PINBALL_LEVELS is not None:
        return APPLICATION_PINBALL_LEVELS
    return PINBALL_LEVELS


def _sample_region_mask(data, q05, q95):
    """Bulk/tail split of the test SAMPLES, on the LAST OBSERVED lag X_t.

    Splitting by the realized target makes every per-region score improper (the
    forecaster's dilemma): on the tail subset the best score goes to whoever
    forecasts the most extreme, not to whoever forecasts honestly. X_t is
    measurable w.r.t. the forecast-time information set, so the subset is the
    same for every method and the region scores stay proper. 'outside' therefore
    reads as "forecasting FROM an extreme state", not "an extreme outcome
    happened". The band itself is unchanged: the theoretical marginal [5%, 95%]
    in simulations, the train+calibration [10%, 90%] on real series.
    """
    x_last = data["X_test"].detach().cpu().numpy()[:, -1].astype(np.float64)
    return (x_last >= q05) & (x_last <= q95)


def _density_test_metrics(recalibrated_density, data, grid_y, q05, q95):
    """Compute the full test-metric dict from a recalibrated predictive density.

    Single source of truth shared by the single-seed and multi-seed paths, and by
    every method (MDN backbones and the CDE baselines). Every metric is reported
    over the same three regions, which partition the test SAMPLES by where the
    last observed lag lands relative to the band (see _sample_region_mask). The
    prediction region is always the (1 - alpha) highest-density
    region (_hdr_regions) -- no equal-tailed / bimodality-test branching.
    """
    y_full = data["y_test"].detach().cpu().numpy().astype(np.float64)
    dens_full = np.asarray(recalibrated_density, dtype=np.float64)

    # The closed-form true density is built on the FULL test split first:
    true_density = None
    if DGP["closed_form"]:
        if DGP["kind"] == "ms_bubble":
            cond_idx = _test_conditioning_indices(data)
            true_density = DGP["model"].predictive_density(
                SERIES, cond_idx, HORIZON, grid_y
            )
        else:
            y_cond_test = data["X_test"].detach().cpu().numpy()[:, -1]  # Y_{t-h}
            true_density = cauchy_ar1_predictive_density(
                grid_x=y_cond_test,
                grid_y=grid_y,
                psi=DGP["noncausal"][0],
                sigma=SIGMA,
                h=HORIZON,
            )
        true_density = np.asarray(true_density, dtype=np.float64)

    # Drop test cases whose realized target lies outside the evaluation grid
    keep = (y_full >= float(grid_y[0])) & (y_full <= float(grid_y[-1]))
    n_test = int(keep.size)
    n_dropped = int((~keep).sum())
    if n_dropped:
        logger.info(
            f"dropping {n_dropped}/{n_test} test targets outside the grid "
            f"({100 * n_dropped / max(n_test, 1):.2f}%)"
        )
    keep_t = torch.as_tensor(keep)
    data = {"y_test": data["y_test"][keep_t], "X_test": data["X_test"][keep_t]}
    y_test_np = y_full[keep]
    recalibrated_density = dens_full[keep]
    if true_density is not None:
        true_density = true_density[keep]

    inside_sample_mask = _sample_region_mask(data, q05, q95)

    metrics = {}

    # Interval metrics computed on three 95% interval constructions:
    #   (default keys) highest-density region (HDR, possibly disjoint),
    #   *_eqt         equal-tailed [q2.5, q97.5],
    #   *_short       shortest connected interval.
    cdf = _cdf_from_density(recalibrated_density, grid_y)
    ci_constructions = (
        ("", _hdr_regions(recalibrated_density, grid_y, alpha=0.05)),
        ("_eqt", _equal_tailed_regions(cdf, grid_y, alpha=0.05)),
        ("_short", _shortest_regions(cdf, grid_y, alpha=0.05)),
    )
    for tag, regions in ci_constructions:
        metrics[f"coverage{tag}"] = _coverage_regions(
            regions, y_test_np, inside_sample_mask
        )
        metrics[f"ci_length{tag}"] = _length_regions(regions, inside_sample_mask)
        metrics[f"winkler{tag}"] = _winkler_regions(
            regions, y_test_np, inside_sample_mask, alpha=0.05
        )

    # Quantile score: total only (see sample_total_mean), and upper levels only
    # for series with no meaningful lower tail.
    levels = _pinball_levels()
    q_pred = _quantiles_from_cdf(cdf, grid_y, levels)
    for j, level in enumerate(levels):
        key = f"pinball_q{int(round(level * 100)):02d}"
        metrics[key] = _sample_total_mean(
            _pinball_samples(q_pred[:, j], y_test_np, level)
        )

    # Point-prediction MSE: mode / mean / median of the recalibrated density
    point_est = _point_estimates(recalibrated_density, grid_y)
    # Only the ratios to the no-change benchmark are reported, so the absolute
    # errors stay local.
    mse_nc = _no_change_mse(data, y_test_np, inside_sample_mask)
    mae_nc = _no_change_mae(data, y_test_np, inside_sample_mask)
    for stat in ("mode", "mean", "median"):
        metrics[f"mse_{stat}_rel"] = _relative_mse(
            _mse_regions(point_est[stat], y_test_np, inside_sample_mask), mse_nc
        )
        metrics[f"mae_{stat}_rel"] = _relative_mae(
            _mae_regions(point_est[stat], y_test_np, inside_sample_mask), mae_nc
        )

    # Density metrics vs the closed-form true density.
    if true_density is not None:
        metrics["ise"] = _ise_regions(
            recalibrated_density, true_density, grid_y, inside_sample_mask
        )
        metrics["hellinger"] = _hellinger_regions(
            recalibrated_density, true_density, grid_y, inside_sample_mask
        )
        metrics["kl"] = _kl_regions(
            recalibrated_density, true_density, grid_y, inside_sample_mask
        )

    return metrics


def evaluate_backbone(backbone, params, q05, q95):
    """Single-seed test evaluation of the tuned best config.

    Trains `params` once at SEED (or reuses the model already trained in this
    process) and reports one set of metrics. Kept for the single-model path; the
    main flow uses evaluate_backbone_multiseed.
    """
    logger.info(f"\n{'=' * 60}\nEVALUATE ON TEST SET: {backbone}\n{'=' * 60}")

    lags = params["lags"]
    d = backbone_dir(backbone)

    if RESCORE_ONLY:
        _require_test_density_cache(d)
        _, per_seed = _rescore_from_cache(d, q05, q95)
        metrics = per_seed[0]
        raw_frames = test_frames = cali_frames = None
    elif REPLAY_CACHED and _has_density_cache(d):
        _, per_seed, raw_frames, test_frames, cali_frames = _replay_from_cache(
            d, q05, q95
        )
        metrics = per_seed[0]
    else:
        if REPLAY_CACHED:
            _log_replay_fallback(d)
        data = get_data(lags)
        grid_y = _build_eval_grid(data, q05, q95)
        mdn = get_density_model(backbone, params)
        artifacts = {}
        _, recalibrated_density = _recalibrate_test(mdn, data, grid_y, artifacts)
        metrics = _density_test_metrics(recalibrated_density, data, grid_y, q05, q95)
        raw_frames, test_frames, cali_frames = _density_frames(
            SEED, data, grid_y, recalibrated_density, artifacts
        )
    if test_frames is not None:  # --rescore leaves the parquets untouched
        _save_density_parquets(d, raw_frames, test_frames, cali_frames)

    logger.info(
        f"[{backbone}] lags={lags} | "
        f"cover(all)={format_number_4_digits(metrics['coverage']['all'])} | "
        f"len(all)={format_number_4_digits(metrics['ci_length']['all'])} | "
        f"winkler(all)={format_number_4_digits(metrics['winkler']['all'])}"
    )
    if DGP["closed_form"]:
        logger.info(
            f"[{backbone}] ISE(all)={metrics['ise']['all']:.5f} | "
            f"H^2(all)={metrics['hellinger']['all']:.5f} | "
            f"KL(all)={metrics['kl']['all']:.5f}"
        )

    _write_json(
        d / "test_eval.json",
        {
            "backbone": backbone,
            "process": DGP["name"],
            "lags": lags,
            "metrics": metrics,
        },
    )
    logger.info(f"[{backbone}] test metrics saved to {d}/test_eval.json")

    return metrics


def _retrain_for_seed(backbone, params, seed):
    """Redraw the series under ``seed`` and retrain the tuned config.

    Returns (mdn, data). Mutates the SERIES / _DATA_CACHE globals; the two set_seed
    calls keep the draw and the training deterministic.
    """
    build_series(PROCESS, seed=seed)
    _DATA_CACHE.clear()
    data = get_data(params["lags"])

    set_seed(seed)  # deterministic init + training RNG for this seed
    mdn = BUILDERS[backbone](params, n_jobs=n_process)
    mdn.fit(
        X_train=data["X_train"],
        y_train=data["y_train"],
        X_val=data["X_cali"],
        y_val=data["y_cali"],
        weights_train=data["weights_train"],
        weights_val=data["weights_val"],
        max_epochs=MAX_EPOCHS,
        learning_rate=params["learning_rate"],
        batch_size=BATCH_SIZE,
        max_norm=MAX_NORM,
        patience=PATIENCE,
        patience_scheduler=PATIENCE_SCHEDULER,
        factor_scheduler=FACTOR_SCHEDULER,
        weighting=WEIGHTING,
    )
    return mdn, data


def _log_seed_metrics(log_tag, seed, metrics):
    """One-line per-seed summary of the headline test metrics."""
    line = (
        f"{log_tag} seed {seed} | "
        f"cover={format_number_4_digits(metrics['coverage']['all'])} | "
        f"winkler={format_number_4_digits(metrics['winkler']['all'])}"
    )
    if DGP["closed_form"]:
        line += f" | ISE={format_number_4_digits(metrics['ise']['all'])}"
    logger.info(line)


def _evaluate_density_multiseed(
    *,
    header,
    log_tag,
    json_key,
    model_label,
    out_dir,
    lags,
    q05,
    q95,
    seeds,
    fit_and_recalibrate,
    point=False,
    extra=None,
):
    """Shared multi-seed density-evaluation loop.

    Used by the MDN backbones and every CDE baseline: per seed, obtain a
    recalibrated density via ``fit_and_recalibrate`` and score it.
    """
    logger.info(
        f"\n{'=' * 60}\nMULTI-SEED TEST EVAL: {header} | "
        f"seeds={list(seeds)}\n{'=' * 60}"
    )

    if RESCORE_ONLY:
        _require_test_density_cache(out_dir)
        seeds, per_seed = _rescore_from_cache(out_dir, q05, q95)
        raw_frames = test_frames = cali_frames = None
        for s, metrics in zip(seeds, per_seed):
            _log_seed_metrics(log_tag, s, metrics)
    elif REPLAY_CACHED and _has_density_cache(out_dir):
        seeds, per_seed, raw_frames, test_frames, cali_frames = _replay_from_cache(
            out_dir, q05, q95
        )
        for s, metrics in zip(seeds, per_seed):
            _log_seed_metrics(log_tag, s, metrics)
    else:
        if REPLAY_CACHED:
            _log_replay_fallback(out_dir)
        per_seed = []
        raw_frames = []
        test_frames = []
        cali_frames = []

        for i, s in enumerate(seeds):
            logger.info(f"{log_tag} seed {s} ({i + 1}/{len(seeds)}): fit + evaluate")
            data, grid_y, recalibrated_density, artifacts = fit_and_recalibrate(s)

            metrics = _density_test_metrics(
                recalibrated_density, data, grid_y, q05, q95
            )
            artifacts.pop("val_loss", None)
            per_seed.append(metrics)

            raw, test, cali = _density_frames(
                s, data, grid_y, recalibrated_density, artifacts
            )
            if raw is not None:
                raw_frames.append(raw)
            test_frames.append(test)
            cali_frames.append(cali)

            _log_seed_metrics(log_tag, s, metrics)

        raw_frames = raw_frames or None

    if test_frames is not None:  # --rescore leaves the parquets untouched
        _save_density_parquets(out_dir, raw_frames, test_frames, cali_frames)

    agg = _aggregate_over_seeds(per_seed)
    _write_multiseed_eval_jsons(
        out_dir,
        process_name=DGP["name"],
        json_key=json_key,
        model_label=model_label,
        lags=lags,
        seeds=seeds,
        agg=agg,
        log_tag=log_tag,
        point=point,
        extra=extra,
    )
    return agg


def evaluate_backbone_multiseed(backbone, params, q05, q95, seeds):
    """Retrain and evaluate the tuned config once per seed.

    Tuning is single-seed; the robustness comes from here. Each seed redraws the
    series and retrains.
    """

    def fit_and_recalibrate(seed):
        mdn, data = _retrain_for_seed(backbone, params, seed)
        grid_y = _build_eval_grid(data, q05, q95)
        artifacts = {}
        val_loss = getattr(mdn, "last_fit_val_loss", None)
        if val_loss is not None:
            artifacts["val_loss"] = val_loss
        _, recalibrated_density = _recalibrate_test(mdn, data, grid_y, artifacts)
        return data, grid_y, recalibrated_density, artifacts

    agg = _evaluate_density_multiseed(
        header=f"{backbone} ({DENSITY})",
        log_tag=f"[{backbone}]",
        json_key="backbone",
        model_label=backbone,
        out_dir=backbone_dir(backbone),
        lags=params["lags"],
        q05=q05,
        q95=q95,
        seeds=seeds,
        fit_and_recalibrate=fit_and_recalibrate,
        point=(DENSITY == "skewt"),
    )

    return agg


# Point-head tuning: given an existing density-optimal config, tune ONLY the
# regression point_head, with the backbone frozen.
_point_head_paths = ArtifactPaths(
    lambda backbone: _algo_dir_impl(OUT_DIR, "point_head")
)
_point_head_dir = _point_head_paths.dir


def _point_head_config_path(backbone):
    return _point_head_paths.config_path(backbone, filename="point_head_config.yaml")


def _point_head_study_db_url(backbone):
    return _point_head_paths.study_db_url(backbone, filename="point_head_study.db")


def load_backbone_for_point_head(backbone, density_params, point_head_layers, n_jobs):
    """Build the model and copy the frozen backbone and density heads.

    Only those weights are copied; the point_head is freshly initialised because its
    architecture differs.
    """
    params = {
        **density_params,
        "point_head_layers": point_head_layers,
        "ph_augmented": True,
    }
    mdn = BUILDERS[backbone](params, n_jobs=n_jobs)

    trained = get_density_model(backbone, density_params)
    backbone_sd = {
        k: v for k, v in trained.state_dict().items() if not k.startswith("point_head")
    }
    missing, unexpected = mdn.load_state_dict(backbone_sd, strict=False)
    assert all(k.startswith("point_head") for k in missing), (
        f"[{backbone}] unexpected missing (non-point_head) keys: "
        f"{[k for k in missing if not k.startswith('point_head')]}"
    )
    assert not unexpected, (
        f"[{backbone}] unexpected keys in density model: {unexpected}"
    )

    mdn.scaler_x = trained.scaler_x
    mdn.scaler_y = trained.scaler_y
    mdn.to(device)
    return mdn


def _fit_and_score_point_head(mdn, data, ph_lr, trial):
    """Train the point_head with the backbone frozen; return the validation MSE."""
    try:
        mdn.fit_point_head(
            X_train=data["X_train"],
            y_train=data["y_train"],
            learning_rate=ph_lr,
            batch_size=BATCH_SIZE,
            max_norm=MAX_NORM,
            patience=PATIENCE,
            patience_scheduler=PATIENCE_SCHEDULER,
            factor_scheduler=FACTOR_SCHEDULER,
            max_epochs=MAX_EPOCHS,
            loss_fn=torch.nn.MSELoss(),
            X_val=data["X_cali"],
            y_val=data["y_cali"],
        )
    except Exception as e:
        logger.warning(f"Point-head trial {trial.number} failed: {e}")
        return float("inf")

    # Objective: validation MSE on the original scale.
    pred = mdn.pred_point(data["X_cali"]).detach().cpu().numpy().astype(np.float64)
    y_val = data["y_cali"].detach().cpu().numpy().astype(np.float64)
    mse = float(np.mean((pred - y_val) ** 2))
    if not np.isfinite(mse):
        return float("inf")
    n_ph = sum(p.numel() for p in mdn.point_head.parameters())
    trial.set_user_attr("n_point_head_params", n_ph)
    return mse


def _point_head_layers(params):
    """Return [ph_width] * ph_depth.

    ``ph_width`` is absent when ph_depth = 0: a bare linear head has no width.
    """
    depth = params["ph_depth"]
    return [params["ph_width"]] * depth if depth > 0 else []


def save_point_head_config(
    backbone, density_params, ph_params, val_mse=None, n_params=None
):
    """Persist the optimal point-head config to point_head_config.yaml."""
    cfg = {
        "backbone": backbone,
        "point_head": {
            "ph_depth": _to_native(ph_params["ph_depth"]),
            "ph_width": _to_native(ph_params.get("ph_width")),
            "ph_learning_rate": _to_native(ph_params["ph_learning_rate"]),
            "point_head_layers": _point_head_layers(ph_params),
        },
        "density_params": {k: _to_native(v) for k, v in density_params.items()},
        "fixed": {
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "max_norm": MAX_NORM,
            "patience": PATIENCE,
            "patience_scheduler": PATIENCE_SCHEDULER,
            "factor_scheduler": FACTOR_SCHEDULER,
            "loss": "mse",
        },
        "val": {
            "mse": _to_native(val_mse),
            "n_point_head_params": _to_native(n_params),
        },
    }
    path = _write_yaml(_point_head_config_path(backbone), cfg)
    logger.info(f"[{backbone}] point-head config saved to {path}")
    return path


def run_point_head_study(backbone, density_params, n_trials, minutes_per_backbone):
    """Optuna study over the point_head, with the density backbone frozen.

    Requires the tuned density config; the backbone it freezes is trained
    in-process by get_density_model.
    """
    if not _config_path(backbone).exists():
        raise SystemExit(
            f"[{backbone}] no density config at {_config_path(backbone)}; "
            f"run density tuning first (e.g. --n_trials N)."
        )

    data = get_data(density_params["lags"])

    def objective(trial):
        set_seed(SEED)
        s = {
            "ph_depth": trial.suggest_int("ph_depth", 0, 2),
            "ph_learning_rate": trial.suggest_categorical(
                "ph_learning_rate",
                [1e-4, 2e-4, 4e-4, 6e-4, 8e-4, 1e-3, 2e-3, 4e-3, 6e-3, 8e-3, 1e-2],
            ),
        }
        # Conditional: a bare linear head (ph_depth = 0) has no width
        if s["ph_depth"] > 0:
            s["ph_width"] = trial.suggest_categorical("ph_width", [16, 32, 64])
        mdn = load_backbone_for_point_head(
            backbone, density_params, _point_head_layers(s), n_jobs=1
        )
        return _fit_and_score_point_head(mdn, data, s["ph_learning_rate"], trial)

    study = run_optuna_study(
        study_name=f"mdn_{backbone}_{DENSITY}_point_head",
        label=f"{backbone} ({DENSITY})",
        header_prefix="TUNING POINT HEAD",
        progress_tag=f"[{backbone}] point-head:",
        out_dir=_point_head_dir(backbone),
        storage=_point_head_study_db_url(backbone),
        tracker_key="point_head",
        objective=objective,
        n_trials=n_trials,
        timeout_minutes=minutes_per_backbone,
        seed=SEED,
    )

    best = study.best_trial
    ph_layers = _point_head_layers(best.params)
    logger.info(
        f"\n[{backbone}] best point-head val MSE: {best.value:.5f} | "
        f"point_head_layers={ph_layers} | lr={best.params['ph_learning_rate']:.3e}"
    )

    save_point_head_config(
        backbone,
        density_params,
        best.params,
        val_mse=best.value,
        n_params=best.user_attrs.get("n_point_head_params"),
    )

    return study


def get_point_head_model(backbone, density_params, ph_params, data=None):
    """Return the tuned point head at SEED, trained on first use and cached."""
    key = (backbone, _params_key(density_params), _params_key(ph_params))
    if key in _TRAINED_POINT_HEAD:
        return _TRAINED_POINT_HEAD[key]

    ph_layers = _point_head_layers(ph_params)
    ckpt_params = {"density": density_params, "point_head": ph_params}
    w = _point_head_weights_path(_point_head_dir(backbone))
    if not FORCE_REFIT and w.exists():
        mdn = BUILDERS[backbone](
            {
                **density_params,
                "point_head_layers": ph_layers,
                "ph_augmented": True,
            },
            n_jobs=n_process,
        )
        if _load_model_weights(w, mdn, ckpt_params):
            logger.info(
                f"[{backbone}] point-head weights restored from {w} -- no training"
            )
            mdn.to(device)
            _TRAINED_POINT_HEAD[key] = mdn
            return mdn

    logger.info(f"[{backbone}] training the best point-head...")
    set_seed(SEED)
    if data is None:
        data = get_data(density_params["lags"])
    mdn = load_backbone_for_point_head(
        backbone,
        density_params,
        _point_head_layers(ph_params),
        n_jobs=n_process,
    )

    class _FauxTrial:
        number = -1

        def set_user_attr(self, *args, **kwargs):
            pass

    _fit_and_score_point_head(mdn, data, ph_params["ph_learning_rate"], _FauxTrial())
    _save_model_weights(w, mdn, ckpt_params)
    _TRAINED_POINT_HEAD[key] = mdn
    return mdn


_PH_CONFIG_KEYS = ("ph_depth", "ph_width", "ph_learning_rate")


def load_point_head_config(backbone):
    """Load point_head_config.yaml and return (density_params, ph_params)."""
    path = _point_head_config_path(backbone)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    ph_params = {
        k: v
        for k, v in cfg["point_head"].items()
        if k in _PH_CONFIG_KEYS and v is not None
    }
    return dict(cfg["density_params"]), ph_params


def evaluate_point_head(backbone, q05, q95):
    """Score the tuned point_head on the test set.

    Same three sample regions as the density-derived point estimates.
    """
    logger.info(
        f"\n{'=' * 60}\nEVALUATE POINT HEAD ON TEST SET: {backbone}\n{'=' * 60}"
    )

    density_params, ph_params = load_point_head_config(backbone)
    lags = density_params["lags"]
    d = _point_head_dir(backbone)

    if RESCORE_ONLY:
        _require_point_cache(d)
        _, per_seed = _rescore_point_from_cache(d, q05, q95)
        m = per_seed[0]
        mse_rel, mae_rel = m["mse_rel"], m["mae_rel"]
    else:
        data = get_data(lags)
        mdn = get_point_head_model(backbone, density_params, ph_params, data=data)
        pred = mdn.pred_point(data["X_test"]).detach().cpu().numpy().astype(np.float64)
        y_test_np = data["y_test"].detach().cpu().numpy().astype(np.float64)
        m = _point_region_metrics(pred, y_test_np, data, q05, q95)
        mse_rel, mae_rel = m["mse_rel"], m["mae_rel"]
        _save_point_parquet(
            d,
            _point_predictions_frame(SEED, y_test_np, _np_last_lag(data), pred),
        )

    logger.info(
        f"[{backbone}] point-head relative MSE (vs no-change) | "
        f"all={mse_rel['all']:.5f} | inside={mse_rel['inside']:.5f} | "
        f"outside={mse_rel['outside']:.5f}"
    )

    _write_json(
        d / "point_head_eval.json",
        {
            "backbone": backbone,
            "lags": lags,
            "mse_point_head_rel": mse_rel,
            "mae_point_head_rel": mae_rel,
        },
    )
    logger.info(f"[{backbone}] point-head metrics saved to {d}/point_head_eval.json")


def evaluate_point_head_multiseed(backbone, density_params, ph_params, q05, q95, seeds):
    """Multi-seed test MSE of the tuned point_head, reported as mean +/- std.

    Apples-to-apples with the density-derived mode/mean/median baselines (which
    are themselves retrained per seed in evaluate_backbone_multiseed): for each
    seed we redraw the series, retrain the backbone + density heads from scratch
    (with the tuned point_head architecture in place), then train that point_head
    on the frozen backbone -- so the head sits on a backbone trained on the same
    realization, not the single SEED model. The point-head *architecture* is
    still selected single-seed by run_point_head_study; only the evaluation is
    multi-seed.
    """
    ph_layers = _point_head_layers(ph_params)
    ph_lr = ph_params["ph_learning_rate"]
    build_params = {
        **density_params,
        "point_head_layers": ph_layers,
        "ph_augmented": True,
    }

    logger.info(
        f"\n{'=' * 60}\nMULTI-SEED POINT-HEAD EVAL: {backbone} ({DENSITY}) | "
        f"point_head_layers={ph_layers} | seeds={list(seeds)}\n{'=' * 60}"
    )

    d = _point_head_dir(backbone)

    if RESCORE_ONLY:
        _require_point_cache(d)
        seeds, cached = _rescore_point_from_cache(d, q05, q95)
        per_seed = [_point_head_metric_keys(m) for m in cached]
        for s, m in zip(seeds, per_seed):
            _log_point_head_seed(backbone, s, m)
        agg_all = _aggregate_over_seeds(per_seed)
        _write_json(
            d / "point_head_eval.json",
            {
                "backbone": backbone,
                "lags": density_params["lags"],
                "point_head_layers": ph_layers,
                "seeds": list(seeds),
                **agg_all,
            },
        )
        logger.info(
            f"[{backbone}] aggregated point-head metrics over {len(seeds)} seeds "
            f"saved to {d}/point_head_eval.json"
        )
        return agg_all

    per_seed, point_frames = [], []
    for i, s in enumerate(seeds):
        logger.info(
            f"[{backbone}] seed {s} ({i + 1}/{len(seeds)}): retrain backbone + head"
        )
        mdn, data = _retrain_for_seed(backbone, build_params, s)

        set_seed(s)
        mdn.fit_point_head(
            X_train=data["X_train"],
            y_train=data["y_train"],
            learning_rate=ph_lr,
            batch_size=BATCH_SIZE,
            max_norm=MAX_NORM,
            patience=PATIENCE,
            patience_scheduler=PATIENCE_SCHEDULER,
            factor_scheduler=FACTOR_SCHEDULER,
            max_epochs=MAX_EPOCHS,
            loss_fn=torch.nn.MSELoss(),
            X_val=data["X_cali"],
            y_val=data["y_cali"],
        )
        pred = mdn.pred_point(data["X_test"]).detach().cpu().numpy().astype(np.float64)
        y_test_np = data["y_test"].detach().cpu().numpy().astype(np.float64)
        m = _point_head_metric_keys(
            _point_region_metrics(pred, y_test_np, data, q05, q95)
        )
        per_seed.append(m)
        point_frames.append(
            _point_predictions_frame(s, y_test_np, _np_last_lag(data), pred)
        )
        _log_point_head_seed(backbone, s, m)

    _save_point_parquet(d, point_frames)

    agg_all = _aggregate_over_seeds(per_seed)
    _write_json(
        d / "point_head_eval.json",
        {
            "backbone": backbone,
            "lags": density_params["lags"],
            "point_head_layers": ph_layers,
            "seeds": list(seeds),
            **agg_all,
        },
    )
    logger.info(
        f"[{backbone}] aggregated point-head metrics over {len(seeds)} seeds "
        f"saved to {d}/point_head_eval.json"
    )

    return agg_all


# FlexZBoost baseline.
_flexzboost_paths = ArtifactPaths(lambda: _algo_dir("flexzboost"))
_flexzboost_dir = _flexzboost_paths.dir
_flexzboost_study_db_url = _flexzboost_paths.study_db_url
_flexzboost_config_path = _flexzboost_paths.config_path


def objective_flexzboost(trial):
    """Optuna objective for the FlexZBoost baseline."""
    set_seed(SEED)
    # FlexCode CDE hyperparameters
    lags = suggest_lags(trial)
    max_basis = FLEXZBOOST_MAX_BASIS
    basis_system = trial.suggest_categorical(
        "basis_system", ["cosine", "Fourier", "db4"]
    )
    # XGBoost regressor hyperparameters
    n_estimators = trial.suggest_int("n_estimators", 100, 2000, log=True)
    max_depth = trial.suggest_int("max_depth", 2, 10, step=2)
    learning_rate = trial.suggest_categorical("learning_rate", FLEXZBOOST_LR_GRID)
    params = {
        "lags": lags,
        "max_basis": max_basis,
        "basis_system": basis_system,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
    }

    data = get_data(lags)
    z_min, z_max = flexzboost_grid_bounds(data)

    try:
        fz = build_and_fit_flexzboost(
            params, data, z_min, z_max, n_jobs=n_process, n_grid=N_GRID
        )
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            val_loss = float(
                fz.estimate_error(
                    data["X_cali"].detach().cpu().numpy(),
                    data["y_cali"].detach().cpu().numpy(),
                    n_grid=N_GRID,
                )
            )
    except Exception as e:
        logger.warning(f"FlexZBoost trial {trial.number} failed: {e}")
        return float("inf")

    if not np.isfinite(val_loss):
        return float("inf")

    trial.set_user_attr("n_best_basis", int(len(fz.best_basis)))
    return val_loss


def save_flexzboost_config(params, val_loss=None, n_best_basis=None):
    """Persist the optimal FlexZBoost config to <flexzboost>/config.yaml."""
    cfg = {
        "model": "flexzboost",
        "params": {
            "lags": _to_native(params["lags"]),
            "max_basis": _to_native(params["max_basis"]),
            "basis_system": params["basis_system"],
            "n_estimators": _to_native(params["n_estimators"]),
            "max_depth": _to_native(params["max_depth"]),
            "learning_rate": _to_native(params["learning_rate"]),
        },
        "fixed": {
            "horizon": HORIZON,
            "regression_model": "XGBoost",
            "xgb_objective": "reg:squarederror",
            "n_grid": N_GRID,
            "bump_threshold_grid": (
                None
                if FLEXZBOOST_BUMP_GRID is None
                else [float(x) for x in FLEXZBOOST_BUMP_GRID]
            ),
            "sharpen_grid": (
                None
                if FLEXZBOOST_SHARPEN_GRID is None
                else [float(x) for x in FLEXZBOOST_SHARPEN_GRID]
            ),
        },
        "val": {
            "cde_loss": _to_native(val_loss),
            "n_best_basis": _to_native(n_best_basis),
        },
    }
    path = _write_yaml(_flexzboost_config_path(), cfg)
    logger.info(f"[flexzboost] optimal config saved to {path}")
    return path


def load_flexzboost_config():
    """Load <flexzboost>/config.yaml and return the params dict the builder expects."""
    path = _flexzboost_config_path()
    with open(path) as f:
        cfg = yaml.safe_load(f)
    params = dict(cfg["params"])
    logger.info(
        f"[flexzboost] config loaded from {path} (lags={params['lags']}, "
        f"max_basis={params['max_basis']}, basis_system={params['basis_system']}, "
        f"n_estimators={params.get('n_estimators')}, "
        f"max_depth={params.get('max_depth')}, "
        f"learning_rate={params.get('learning_rate')})"
    )
    return params


def run_flexzboost_study(n_trials, minutes_per_backbone):
    """Run an Optuna study over the FlexZBoost hyperparameters."""
    out_dir = _flexzboost_dir()
    study = run_optuna_study(
        study_name="flexzboost",
        label="flexzboost (FlexCode+XGBoost)",
        progress_tag="[flexzboost]",
        out_dir=out_dir,
        storage=_flexzboost_study_db_url(),
        tracker_key="flexzboost",
        objective=objective_flexzboost,
        n_trials=n_trials,
        timeout_minutes=minutes_per_backbone,
        seed=SEED,
    )

    best = study.best_trial
    resolved = dict(best.params)
    resolved.setdefault("max_basis", FLEXZBOOST_MAX_BASIS)
    logger.info(f"\n[flexzboost] best val CDE loss: {best.value:.5f}")
    logger.info(
        f"[flexzboost] best params: lags={resolved['lags']} | "
        f"max_basis={resolved['max_basis']} | basis_system={resolved['basis_system']} "
        f"| n_estimators={resolved['n_estimators']} "
        f"| max_depth={resolved['max_depth']} "
        f"| learning_rate={resolved['learning_rate']} "
        f"| n_best_basis={best.user_attrs.get('n_best_basis', 'NA')}"
    )

    save_flexzboost_config(
        resolved, val_loss=best.value, n_best_basis=best.user_attrs.get("n_best_basis")
    )
    return study


def evaluate_flexzboost_multiseed(params, q05, q95, seeds):
    """Refit and evaluate the tuned FlexZBoost config once per seed."""
    lags = params["lags"]

    def fit_and_recalibrate(seed):
        build_series(PROCESS, seed=seed)
        _DATA_CACHE.clear()
        data = get_data(lags)
        set_seed(seed)

        grid_y = _build_eval_grid(data, q05, q95)
        z_min, z_max = float(grid_y[0]), float(grid_y[-1])
        fz = build_and_fit_flexzboost(
            params, data, z_min, z_max, n_jobs=n_process, n_grid=N_GRID
        )
        artifacts = {}
        _, recalibrated_density = recalibrate_test_flexzboost(
            fz, data, grid_y, N_GRID, device, n_process, artifacts=artifacts
        )
        return data, grid_y, recalibrated_density, artifacts

    return _evaluate_density_multiseed(
        header="flexzboost (FlexCode+XGBoost)",
        log_tag="[flexzboost]",
        json_key="model",
        model_label="flexzboost",
        out_dir=_flexzboost_dir(),
        lags=lags,
        q05=q05,
        q95=q95,
        seeds=seeds,
        fit_and_recalibrate=fit_and_recalibrate,
    )


# Kernel conditional density estimation baseline.
_kcde_paths = ArtifactPaths(lambda: _algo_dir("kcde"))
_kcde_dir = _kcde_paths.dir
_kcde_study_db_url = _kcde_paths.study_db_url
_kcde_config_path = _kcde_paths.config_path


def objective_kcde(trial):
    """Optuna objective for the KCDE baseline."""
    set_seed(SEED)

    lags = suggest_lags(trial)
    bandwidth_x = trial.suggest_categorical("bandwidth_x", KCDE_BANDWIDTH_GRID)
    bandwidth_y = trial.suggest_categorical("bandwidth_y", KCDE_BANDWIDTH_GRID)
    kernel = trial.suggest_categorical("kernel", KCDE_KERNELS)

    params = {
        "lags": lags,
        "bandwidth_x": bandwidth_x,
        "bandwidth_y": bandwidth_y,
        "kernel": kernel,
    }

    data = get_data(lags)
    grid_y = kcde_grid(data, N_GRID)

    try:
        val_loss = kcde_val_loss(params, data, grid_y)
    except Exception as e:
        logger.warning(f"KCDE trial {trial.number} failed: {e}")
        return float("inf")

    if not np.isfinite(val_loss):
        return float("inf")

    return val_loss


def save_kcde_config(params, val_loss=None):
    """Persist the optimal KCDE config to <kcde>/config.yaml."""
    cfg = {
        "model": "kcde",
        "params": {
            "lags": _to_native(params["lags"]),
            "bandwidth_x": _to_native(params["bandwidth_x"]),
            "bandwidth_y": _to_native(params["bandwidth_y"]),
            "kernel": params["kernel"],
        },
        "fixed": {
            "horizon": HORIZON,
            "n_grid": N_GRID,
            "bandwidth_grid": [float(x) for x in KCDE_BANDWIDTH_GRID],
            "kernels": KCDE_KERNELS,
            "scaler": "RobustScaler",
        },
        "val": {
            "cde_nll": _to_native(val_loss),
        },
    }

    path = _write_yaml(_kcde_config_path(), cfg)
    logger.info(f"[kcde] optimal config saved to {path}")
    return path


def load_kcde_config():
    """Load <kcde>/config.yaml and return the params dict."""
    path = _kcde_config_path()
    with open(path) as f:
        cfg = yaml.safe_load(f)

    params = dict(cfg["params"])
    logger.info(
        f"[kcde] config loaded from {path} "
        f"(lags={params['lags']}, bandwidth_x={params['bandwidth_x']}, "
        f"bandwidth_y={params['bandwidth_y']}, kernel={params['kernel']})"
    )
    return params


def run_kcde_study(n_trials, minutes_per_backbone):
    """Optuna study over lags, bandwidth_x, bandwidth_y, and kernel type."""
    out_dir = _kcde_dir()
    study = run_optuna_study(
        study_name="kcde",
        label="kcde (kernel CDE)",
        progress_tag="[kcde]",
        out_dir=out_dir,
        storage=_kcde_study_db_url(),
        tracker_key="kcde",
        objective=objective_kcde,
        n_trials=n_trials,
        timeout_minutes=minutes_per_backbone,
        seed=SEED,
    )

    best = study.best_trial
    resolved = dict(best.params)

    logger.info(f"\n[kcde] best val CDE NLL: {best.value:.5f}")
    logger.info(
        f"[kcde] best params: lags={resolved['lags']} | "
        f"bandwidth_x={resolved['bandwidth_x']} | "
        f"bandwidth_y={resolved['bandwidth_y']} | kernel={resolved['kernel']}"
    )

    save_kcde_config(resolved, val_loss=best.value)
    return study


def evaluate_kcde_multiseed(params, q05, q95, seeds):
    """Fit + evaluate KCDE once per seed and report mean +/- std."""
    lags = params["lags"]

    def fit_and_recalibrate(seed):
        build_series(PROCESS, seed=seed)
        _DATA_CACHE.clear()
        data = get_data(lags)
        set_seed(seed)

        grid_y = _build_eval_grid(data, q05, q95)
        artifacts = {}
        _, recalibrated_density = recalibrate_test_kcde(
            params, data, grid_y, device, n_process, artifacts=artifacts
        )
        return data, grid_y, recalibrated_density, artifacts

    return _evaluate_density_multiseed(
        header="kcde (kernel CDE)",
        log_tag="[kcde]",
        json_key="model",
        model_label="kcde",
        out_dir=_kcde_dir(),
        lags=lags,
        q05=q05,
        q95=q95,
        seeds=seeds,
        fit_and_recalibrate=fit_and_recalibrate,
    )


# Conditional normalizing-flow baseline (zuko rational-quadratic neural spline flow).


_flow_paths = ArtifactPaths(lambda: _algo_dir("flow"))
_flow_dir = _flow_paths.dir
_flow_study_db_url = _flow_paths.study_db_url
_flow_config_path = _flow_paths.config_path


def objective_flow(trial):
    """Optuna objective for the conditional-flow baseline."""
    set_seed(SEED)

    lags = suggest_lags(trial)
    transforms = trial.suggest_int("transforms", *FLOW_TRANSFORMS_RANGE)
    hidden_depth = trial.suggest_int("hidden_depth", *FLOW_DEPTH_RANGE)
    hidden_width = trial.suggest_categorical("hidden_width", FLOW_WIDTH_GRID)
    bins = trial.suggest_categorical("bins", FLOW_BINS_GRID)
    learning_rate = trial.suggest_categorical("learning_rate", FLOW_LR_GRID)

    params = {
        "lags": lags,
        "transforms": transforms,
        "hidden_depth": hidden_depth,
        "hidden_width": hidden_width,
        "bins": bins,
        "learning_rate": learning_rate,
    }

    data = get_data(lags)

    try:
        flow = build_flow(params, device)
        n_params = sum(p.numel() for p in flow.parameters())
        scalers = fit_flow_scalers(data)
        val_nll = train_flow(
            flow,
            data,
            scalers,
            learning_rate,
            device,
            max_epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            max_norm=MAX_NORM,
            patience=PATIENCE,
            patience_scheduler=PATIENCE_SCHEDULER,
            factor_scheduler=FACTOR_SCHEDULER,
        )
    except Exception as e:
        logger.warning(f"Flow trial {trial.number} failed: {e}")
        return float("inf")

    if not np.isfinite(val_nll):
        return float("inf")

    trial.set_user_attr("n_density_params", int(n_params))
    return val_nll


def save_flow_config(params, val_nll=None, n_params=None):
    """Persist the optimal flow config to <flow>/config.yaml."""
    cfg = {
        "model": "flow",
        "params": {
            "lags": _to_native(params["lags"]),
            "transforms": _to_native(params["transforms"]),
            "hidden_depth": _to_native(params["hidden_depth"]),
            "hidden_width": _to_native(params["hidden_width"]),
            "bins": _to_native(params["bins"]),
            "learning_rate": _to_native(params["learning_rate"]),
        },
        "fixed": {
            "horizon": HORIZON,
            "n_grid": N_GRID,
            "flow": "NSF (rational-quadratic neural spline)",
            "base_distribution": "standard_normal",
            "scaler": "RobustScaler",
            "weighting": "none",
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "patience_scheduler": PATIENCE_SCHEDULER,
            "factor_scheduler": FACTOR_SCHEDULER,
            "batch_size": BATCH_SIZE,
            "max_norm": MAX_NORM,
            "width_grid": list(FLOW_WIDTH_GRID),
            "bins_grid": list(FLOW_BINS_GRID),
        },
        "val": {
            "nll": _to_native(val_nll),
            "n_params": _to_native(n_params),
        },
    }
    path = _write_yaml(_flow_config_path(), cfg)
    logger.info(f"[flow] optimal config saved to {path}")
    return path


def load_flow_config():
    """Load <flow>/config.yaml and return the params dict the builder expects."""
    path = _flow_config_path()
    with open(path) as f:
        cfg = yaml.safe_load(f)
    params = dict(cfg["params"])
    logger.info(
        f"[flow] config loaded from {path} (lags={params['lags']}, "
        f"transforms={params['transforms']}, hidden_depth={params['hidden_depth']}, "
        f"hidden_width={params['hidden_width']}, bins={params['bins']}, "
        f"learning_rate={params['learning_rate']})"
    )
    return params


def run_flow_study(n_trials, minutes_per_backbone):
    """Run an Optuna study over the flow hyperparameters, minimising NLL."""
    out_dir = _flow_dir()
    study = run_optuna_study(
        study_name="flow",
        label="flow (conditional NSF)",
        progress_tag="[flow]",
        out_dir=out_dir,
        storage=_flow_study_db_url(),
        tracker_key="flow",
        objective=objective_flow,
        n_trials=n_trials,
        timeout_minutes=minutes_per_backbone,
        seed=SEED,
    )

    best = study.best_trial
    resolved = dict(best.params)
    logger.info(f"\n[flow] best val NLL: {best.value:.5f}")
    logger.info(
        f"[flow] best params: lags={resolved['lags']} | "
        f"transforms={resolved['transforms']} "
        f"| hidden_depth={resolved['hidden_depth']} "
        f"| hidden_width={resolved['hidden_width']} | bins={resolved['bins']} "
        f"| learning_rate={resolved['learning_rate']} "
        f"| n_params={best.user_attrs.get('n_density_params', 'NA')}"
    )

    save_flow_config(
        resolved,
        val_nll=best.value,
        n_params=best.user_attrs.get("n_density_params"),
    )
    return study


def evaluate_flow_multiseed(params, q05, q95, seeds):
    """Refit and evaluate the tuned flow config once per seed."""
    lags = params["lags"]

    def fit_and_recalibrate(seed):
        build_series(PROCESS, seed=seed)
        _DATA_CACHE.clear()
        data = get_data(lags)
        set_seed(seed)

        grid_y = _build_eval_grid(data, q05, q95)
        flow = build_flow(params, device)
        scalers = fit_flow_scalers(data)
        train_flow(
            flow,
            data,
            scalers,
            params["learning_rate"],
            device,
            max_epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            max_norm=MAX_NORM,
            patience=PATIENCE,
            patience_scheduler=PATIENCE_SCHEDULER,
            factor_scheduler=FACTOR_SCHEDULER,
        )
        artifacts = {}
        _, recalibrated_density = recalibrate_test_flow(
            flow, scalers, data, grid_y, device, n_process, artifacts=artifacts
        )
        return data, grid_y, recalibrated_density, artifacts

    return _evaluate_density_multiseed(
        header="flow (conditional NSF)",
        log_tag="[flow]",
        json_key="model",
        model_label="flow",
        out_dir=_flow_dir(),
        lags=lags,
        q05=q05,
        q95=q95,
        seeds=seeds,
        fit_and_recalibrate=fit_and_recalibrate,
    )


# Conformal-prediction baseline (MAPIE TimeSeriesRegressor: ACI).

_conformal_paths = ArtifactPaths(lambda backbone: _algo_dir("conformal"))
_conformal_dir = _conformal_paths.dir
_conformal_study_db_url = _conformal_paths.study_db_url
_conformal_config_path = _conformal_paths.config_path


def _gamma_tag(gamma):
    """Map an ACI step size to its folder tag: 0.01 -> ``gamma_01``."""
    if gamma == 0:
        return "gamma_0"
    frac = f"{gamma:.10f}".rstrip("0").split(".")[1]
    return f"gamma_{frac}"


def _conformal_gamma_dir(backbone, gamma):
    """Per-gamma evaluation subfolder: <PROCESS>/conformal/gamma_<tag>/.

    The tuned point predictor (config.yaml / study.db) is gamma-independent and
    stays at the conformal/ level; only the per-gamma ACI evaluation JSONs
    (test_eval*.json) land in these subfolders.
    """
    return _algo_dir_impl(_conformal_dir(backbone), _gamma_tag(gamma))


def _fit_mse_unweighted(mdn, data, lr):
    """Train backbone + point_head with fit_mse, fully UNWEIGHTED.

    weighting="none" already disables the sampler and forces unit loss weights;
    unit weight tensors are passed too so the call is unambiguous. Returns the
    best (unweighted) validation MSE.
    """
    return fit_mse_unweighted(
        mdn,
        data,
        lr,
        device,
        max_epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        max_norm=MAX_NORM,
        patience=PATIENCE,
        patience_scheduler=PATIENCE_SCHEDULER,
        factor_scheduler=FACTOR_SCHEDULER,
    )


def _fit_and_score_mse(mdn, data, lr, trial):
    """Train one point predictor (unweighted MSE) and return the validation MSE."""
    try:
        val_mse = _fit_mse_unweighted(mdn, data, lr)
    except Exception as e:  # numerical blow-ups, invalid configs, etc.
        logger.warning(f"Conformal trial {trial.number} failed: {e}")
        return float("inf")
    if not np.isfinite(val_mse):
        return float("inf")
    n_pred_params = sum(
        p.numel()
        for name, p in mdn.named_parameters()
        if not name.startswith(
            ("pi_layer", "mu_layer", "sigma_layer", "nu_layer", "lam_layer")
        )
    )
    trial.set_user_attr("n_predictor_params", n_pred_params)
    return val_mse


def _suggest_shared_conformal(trial):
    """Shared search space for the conformal point predictor.

    Identical to _suggest_shared minus n_mixtures: the density resolution is
    irrelevant once the density heads are frozen.
    """
    return {
        "lags": suggest_lags(trial),
        "learning_rate": suggest_learning_rate(trial),
        "dropout": suggest_dropout(trial),
        "n_mixtures": 1,
    }


def objective_conformal(trial, backbone):
    """Optuna objective for the conformal point predictor."""
    set_seed(SEED)
    s = _suggest_shared_conformal(trial)
    data = get_data(s["lags"])
    s = suggest_backbone_arch(trial, backbone, s)
    mdn = BUILDERS[backbone](s, n_jobs=1)
    return _fit_and_score_mse(mdn, data, s["learning_rate"], trial)


def save_conformal_config(backbone, params, val_mse=None, n_params=None):
    """Persist the optimal point-predictor config to <conformal>/config.yaml."""
    cfg = {
        "backbone": backbone,
        "model": "conformal_point_predictor",
        "params": {k: _to_native(v) for k, v in params.items()},
        "fixed": {
            "horizon": HORIZON,
            "objective": "mse",
            "weighting": "none",
            "batch_size": BATCH_SIZE,
            "point_head_layers": POINT_HEAD_LAYERS,
            "max_epochs": MAX_EPOCHS,
            "max_norm": MAX_NORM,
            "patience": PATIENCE,
            "patience_scheduler": PATIENCE_SCHEDULER,
            "factor_scheduler": FACTOR_SCHEDULER,
            "conformal_alpha": CONFORMAL_ALPHA,
            "aci_gammas": CONFORMAL_ACI_GAMMAS,
        },
        "val": {
            "mse": _to_native(val_mse),
            "n_predictor_params": _to_native(n_params),
        },
    }
    path = _write_yaml(_conformal_config_path(backbone), cfg)
    logger.info(f"[{backbone}] conformal config saved to {path}")
    return path


def load_conformal_config(backbone):
    """Load <conformal>/config.yaml and return (backbone, params)."""
    path = _conformal_config_path(backbone)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    saved_backbone = cfg.get("backbone", backbone)
    params = dict(cfg["params"])
    params.setdefault("n_mixtures", 1)
    logger.info(f"[{saved_backbone}] conformal config loaded from {path}")
    return saved_backbone, params


def run_conformal_study(backbone, n_trials, minutes_per_backbone):
    """Optuna study over the MDN search space, minimising validation MSE.

    Persists study.db / config.yaml; evaluation retrains per seed.
    """
    out_dir = _conformal_dir(backbone)
    study = run_optuna_study(
        study_name=f"conformal_{backbone}",
        label=f"{backbone} conformal point predictor (MSE)",
        header_sep=" ",
        progress_tag=f"[{backbone}] conformal:",
        out_dir=out_dir,
        storage=_conformal_study_db_url(backbone),
        tracker_key="conformal",
        objective=lambda t: objective_conformal(t, backbone),
        n_trials=n_trials,
        timeout_minutes=minutes_per_backbone,
        seed=SEED,
    )

    best = study.best_trial
    resolved = dict(best.params)
    resolved.setdefault("n_mixtures", 1)
    logger.info(f"\n[{backbone}] best val MSE: {best.value:.5f}")
    logger.info(
        f"[{backbone}] predictor params: "
        f"{best.user_attrs.get('n_predictor_params', 'NA')}"
    )

    save_conformal_config(
        backbone,
        resolved,
        val_mse=best.value,
        n_params=best.user_attrs.get("n_predictor_params"),
    )

    return study


def _conformal_regions(mdn, data, aci_gamma, alpha=CONFORMAL_ALPHA):
    """Build the ACI prediction regions for the test set."""
    return conformal_regions(mdn, data, aci_gamma, alpha, SEED, logger=logger)


def _retrain_point_predictor_for_seed(backbone, params, seed):
    """Redraw the series under ``seed`` and retrain the point predictor.

    Mirrors _retrain_for_seed but uses fit_mse / weighting="none".
    """
    build_series(PROCESS, seed=seed)
    _DATA_CACHE.clear()
    data = get_data(params["lags"])

    set_seed(seed)
    mdn = BUILDERS[backbone](params, n_jobs=n_process)
    _fit_mse_unweighted(mdn, data, params["learning_rate"])
    return mdn, data


def evaluate_conformal_multiseed(backbone, params, q05, q95, seeds, gammas):
    """Retrain per seed and recompute the ACI interval for each gamma.

    The point predictor is gamma-independent, so each seed is retrained once.
    """
    logger.info(
        f"\n{'=' * 60}\nMULTI-SEED CONFORMAL EVAL: {backbone} (ACI) | "
        f"seeds={list(seeds)} | gammas={list(gammas)}\n{'=' * 60}"
    )
    lags = params["lags"]
    per_seed = {g: [] for g in gammas}

    for i, s in enumerate(seeds):
        logger.info(
            f"[{backbone}] seed {s} ({i + 1}/{len(seeds)}): retrain + conformalize"
        )
        mdn, data = _retrain_point_predictor_for_seed(backbone, params, s)

        y_test_np = data["y_test"].detach().cpu().numpy().astype(np.float64)
        inside_sample_mask = _sample_region_mask(data, q05, q95)

        for g in gammas:
            regions = _conformal_regions(mdn, data, aci_gamma=g)
            m = _conformal_interval_metrics(
                regions, y_test_np, inside_sample_mask, alpha=CONFORMAL_ALPHA
            )
            per_seed[g].append(m)
            logger.info(
                f"[{backbone}] seed {s} | aci gamma={g} | "
                f"cover={format_number_4_digits(m['coverage']['all'])} | "
                f"len={format_number_4_digits(m['ci_length']['all'])} | "
                f"winkler={format_number_4_digits(m['winkler']['all'])}"
            )

    results = {}
    for g in gammas:
        agg = _aggregate_over_seeds(per_seed[g])
        d = _conformal_gamma_dir(backbone, g)
        _write_multiseed_eval_jsons(
            d,
            process_name=DGP["name"],
            json_key="backbone",
            model_label=backbone,
            lags=lags,
            seeds=seeds,
            agg=agg,
            log_tag=f"[{backbone}] gamma={g}:",
            extra={"aci_gamma": g},
        )
        results[_gamma_tag(g)] = agg

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_trials",
        type=int,
        default=None,
        help="Max trials for the study (required unless --evaluate is passed).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        required=True,
        help="Forecast horizon h (predict Y_t from Y_{t-h}). Sets the "
        "output root outputs/simulations/horizon_<h>/.",
    )
    parser.add_argument(
        "--density",
        choices=[
            "gaussian",
            "student",
            "skewt",
            "skewnorm",
        ],
        default=None,
        help="Mixture component density (the MDN NLL head). Required for "
        "--model mdn. Ignored by the baseline models "
        "(flexzboost/kcde/conformal/flow); --model point_head "
        "defaults it to skewt.",
    )
    parser.add_argument(
        "--weighting",
        choices=["both", "sampler", "loss", "none"],
        default=None,
        help="Tail-weighting scheme for fit(): 'both' (WeightedRandomSampler + "
        "weighted NLL), 'sampler' (sampler only), 'loss' (weighted NLL "
        "only), or 'none'. Required for --model mdn; ignored by the baseline "
        "models and defaulted to 'both' for --model point_head.",
    )
    parser.add_argument(
        "--process",
        required=True,
        help="Data-generating-process tag from configs/dgp_config.yaml.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="If given, flattens the output root to "
        "outputs/simulations/<name>/ instead of "
        "outputs/simulations/<process>/horizon_<h>/.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Skip tuning and re-score from the saved raw_densities.parquet: only "
        "the recalibrator is refitted, so a calibrator change is re-evaluated in "
        "seconds. Falls back to retraining from config.yaml when the parquets are "
        "not on disk yet.",
    )
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="Skip tuning, training AND recalibration: recompute the metrics "
        "straight from the saved test_densities.parquet (the already-calibrated "
        "densities). Use this to re-run a change to the METRICS over finished "
        "runs, where --evaluate re-runs a change to the CALIBRATOR. Implies "
        "--evaluate; falls back to it when the parquet is not on disk.",
    )
    parser.add_argument(
        "--no_save_raw",
        dest="save_raw",
        action="store_false",
        help="Skip raw_densities.parquet (the UNcalibrated densities). They are "
        "saved by default because only they let the recalibrator be refitted "
        "without retraining (--evaluate without --refit); pass this to halve a "
        "cell's footprint when you will not sweep recalibrator settings.",
    )
    parser.add_argument(
        "--refit",
        action="store_true",
        help="With --evaluate, reuse the saved configuration but retrain the model "
        "and recalibrator for every evaluation seed instead of replaying cached "
        "density Parquets.",
    )
    parser.add_argument(
        "--model",
        choices=[
            "mdn",
            "point_head",
            "flexzboost",
            "kcde",
            "conformal",
            "flow",
        ],
        default="mdn",
        help="Which model to tune/evaluate. 'mdn' (default): the MDN density "
        "backbone (uses --density/--weighting). 'point_head': a regression head "
        "on a FROZEN backbone. The rest are baselines "
        "(flexzboost/kcde/conformal/flow) that ignore "
        "--density/--weighting.",
    )
    args = parser.parse_args()

    if args.rescore:
        if args.refit:
            parser.error("--rescore and --refit are mutually exclusive")
        args.evaluate = True
    if args.refit and not args.evaluate:
        parser.error("--refit requires --evaluate")

    args.point_head = args.model == "point_head"
    args.flexzboost = args.model == "flexzboost"
    args.kcde = args.model == "kcde"
    args.conformal = args.model == "conformal"
    args.flow = args.model == "flow"

    _mdn_ablation = args.model == "mdn" and (
        args.density in _SATURATED_FAMILIES
        or (args.density == "skewt" and args.weighting in _WEIGHTING_ABLATION)
    )
    if not args.evaluate and args.n_trials is None and not _mdn_ablation:
        parser.error("--n_trials is required when --evaluate is not passed.")

    if args.point_head:
        if args.density is None:
            args.density = "skewt"
        if args.weighting is None:
            args.weighting = "both"
    if args.model == "mdn":
        if args.density is None:
            parser.error("--density is required for --model mdn.")
        if args.weighting is None:
            parser.error("--weighting is required for --model mdn.")
    if args.point_head and args.density != "skewt":
        parser.error(
            "--model point_head only supports --density skewt "
            f"(got --density {args.density})."
        )

    REPLAY_CACHED = args.evaluate and not args.refit
    RESCORE_ONLY = args.rescore
    FORCE_REFIT = args.refit
    SAVE_RAW = args.save_raw

    init(
        horizon=args.horizon,
        process=args.process,
        density=args.density,
        weighting=args.weighting,
        name=args.name,
    )

    backbones = ["lstm"]

    # Baseline modes.
    def _run_baseline(load_label, tune, load_config, evaluate, report):
        q05, q95 = _theoretical_band()
        if args.evaluate:
            logger.info(f"Loading saved {load_label} config -- skipping tuning.")
            params = load_config()
        else:
            tune()
            params = load_config()
        report(evaluate(params, q05, q95))
        raise SystemExit(0)

    _BASELINES = {
        "conformal": dict(
            active=args.conformal,
            load_label="conformal",
            tune=lambda: run_conformal_study("lstm", args.n_trials, None),
            load_config=lambda: load_conformal_config("lstm")[1],
            evaluate=lambda p, q05, q95: evaluate_conformal_multiseed(
                "lstm", p, q05, q95, EVAL_SEEDS, CONFORMAL_ACI_GAMMAS
            ),
            report=lambda agg: _print_eval_tables(
                agg, EVAL_SEEDS, label_header="gamma"
            ),
        ),
        "kcde": dict(
            active=args.kcde,
            load_label="KCDE",
            tune=lambda: run_kcde_study(args.n_trials, None),
            load_config=load_kcde_config,
            evaluate=lambda p, q05, q95: evaluate_kcde_multiseed(
                p, q05, q95, EVAL_SEEDS
            ),
            report=lambda agg: _print_eval_tables(
                {"kcde": agg}, EVAL_SEEDS, label_header="model"
            ),
        ),
        "flow": dict(
            active=args.flow,
            load_label="flow",
            tune=lambda: run_flow_study(args.n_trials, None),
            load_config=load_flow_config,
            evaluate=lambda p, q05, q95: evaluate_flow_multiseed(
                p, q05, q95, EVAL_SEEDS
            ),
            report=lambda agg: _print_eval_tables(
                {"flow": agg}, EVAL_SEEDS, label_header="model"
            ),
        ),
        "flexzboost": dict(
            active=args.flexzboost,
            load_label="FlexZBoost",
            tune=lambda: run_flexzboost_study(args.n_trials, None),
            load_config=load_flexzboost_config,
            evaluate=lambda p, q05, q95: evaluate_flexzboost_multiseed(
                p, q05, q95, EVAL_SEEDS
            ),
            report=lambda agg: _print_eval_tables(
                {"flexzboost": agg}, EVAL_SEEDS, label_header="model"
            ),
        ),
    }
    for _spec in _BASELINES.values():
        if _spec["active"]:
            _run_baseline(
                _spec["load_label"],
                _spec["tune"],
                _spec["load_config"],
                _spec["evaluate"],
                _spec["report"],
            )

    if args.point_head:
        per_backbone = None
        q05, q95 = _theoretical_band()
        ph_results = {}
        for b in backbones:
            if args.evaluate:
                # Skip tuning: reload the saved point-head config (which also
                # carries the density params it was tuned on) and evaluate.
                ph_cfg_path = _point_head_config_path(b)
                if not ph_cfg_path.exists():
                    raise SystemExit(
                        f"[{b}] no point-head config at {ph_cfg_path}; run "
                        f"--point_head tuning first (without --evaluate)."
                    )
                logger.info("Loading saved point-head config -- skipping tuning.")
                density_params, ph_params = load_point_head_config(b)
            else:
                if not _config_path(b).exists():
                    logger.info(
                        f"[{b}] no density config at {_config_path(b)}; tuning the "
                        f"base model first."
                    )
                    run_study(b, args.n_trials, None)
                density_params = load_config_yaml(b)
                study = run_point_head_study(
                    b, density_params, args.n_trials, per_backbone
                )
                ph_params = study.best_trial.params
            ph_results[b] = evaluate_point_head_multiseed(
                b, density_params, ph_params, q05, q95, EVAL_SEEDS
            )

        cw = 18
        header = (
            f"{'backbone':<12} | {'all':>{cw}} | {'inside[5-95%]':>{cw}} | "
            f"{'outside':>{cw}}"
        )
        ph_tables = (
            (
                "mse_point_head",
                "MSE -- tuned point head (vs realized target; sample regions)",
            ),
            (
                "mse_point_head_rel",
                "Relative MSE -- tuned point head / no-change benchmark "
                "(<1 beats no-change; per-seed ratio)",
            ),
            (
                "mae_point_head",
                "MAE -- tuned point head (vs realized target; sample regions)",
            ),
            (
                "mae_point_head_rel",
                "Relative MAE -- tuned point head / no-change benchmark "
                "(<1 beats no-change; per-seed ratio)",
            ),
        )
        for key, title in ph_tables:
            logger.info("\n" + "=" * len(header))
            logger.info(
                f"{title} -- mean+/-std over {len(EVAL_SEEDS)} seeds {list(EVAL_SEEDS)}"
            )
            logger.info("=" * len(header))
            logger.info(header)
            logger.info("-" * len(header))
            for b, m in ph_results.items():
                cell = m[key]
                logger.info(
                    f"{b:<12} | {_fmt_cell(cell['all']):>{cw}} | "
                    f"{_fmt_cell(cell['inside']):>{cw}} | "
                    f"{_fmt_cell(cell['outside']):>{cw}}"
                )
            logger.info("=" * len(header))
        raise SystemExit(0)

    if _reuses_flagship_config():
        logger.info(
            f"{_ablation_label()} -- reusing skew-t/both config, no Optuna search "
            f"(any --n_trials is ignored)."
        )
        eval_params = {b: load_skewt_both_config(b) for b in backbones}
    elif args.evaluate:
        logger.info("Loading saved config(s) -- skipping tuning.")
        eval_params = {b: load_config_yaml(b) for b in backbones}
    else:
        studies = {b: run_study(b, args.n_trials, None) for b in backbones}

        if len(studies) > 1:
            logger.info(f"\n{'=' * 60}\nSUMMARY (best val NLL)\n{'=' * 60}")
            for b, st in studies.items():
                t = st.best_trial
                logger.info(
                    f"{b:<12} | NLL={t.value:.5f} | "
                    f"params={t.user_attrs.get('n_density_params', 'NA')}"
                )

        eval_params = {b: load_config_yaml(b) for b in backbones}

    # Test-set evaluation of the best model per backbone
    q05, q95 = _theoretical_band()
    eval_results = {
        b: evaluate_backbone_multiseed(b, params, q05, q95, EVAL_SEEDS)
        for b, params in eval_params.items()
    }

    _print_eval_tables(eval_results, EVAL_SEEDS)
