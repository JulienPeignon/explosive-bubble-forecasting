"""Density forecasting on the real Brent series, reusing the simulation pipeline."""

import json
import os
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from matplotlib.patches import ConnectionPatch

# Configuration
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

BRENT_SERIES_ID = "DCOILBRENTEU"  # Daily Brent crude oil spot price
CPI_SERIES_ID = "CPIAUCSL"  # Monthly US CPI (all urban consumers)

BRENT_START_DATE = "2000-01-01"
CPI_START_DATE = "1990-01-01"
END_DATE = "2026-07-01"  # Hard-coded end date

DATA_DIR = "data"
BRENT_PARQUET_PATH = os.path.join(DATA_DIR, "brent.parquet")

PLOTS_DIR = os.path.join("outputs", "applications", "plots")

CONFIG_PATH = os.path.join("configs", "run_config.yaml")


# API key handling
def get_api_key() -> str:
    """Load the FRED API key from the .env file."""
    load_dotenv()
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise RuntimeError("FRED_API_KEY not found. Check your .env file.")
    return api_key


# Config handling
def load_proportions() -> list:
    """
    Read the train/calibration/test split proportions for applications.

    A tiny dependency-free parser is used (only the single `proportions:` list
    is needed) to avoid pulling in a YAML library.
    """
    with open(CONFIG_PATH) as f:
        for line in f:
            line = line.strip()
            if line.startswith("proportions:"):
                # Extract the bracketed list, e.g. "[0.6, 0.2, 0.2]".
                raw = line.split(":", 1)[1].strip().strip("[]")
                props = [float(x) for x in raw.split(",")]
                if abs(sum(props) - 1.0) > 1e-6:
                    raise ValueError(f"Proportions must sum to 1.0, got {props}")
                return props
    raise ValueError(f"No 'proportions' key found in {CONFIG_PATH}")


def split_series(df: pd.DataFrame, proportions: list) -> dict:
    """Split the ordered dataset into train/calibration/test DataFrames."""
    n = len(df)
    train_end = int(n * proportions[0])
    cali_end = train_end + int(n * proportions[1])
    return {
        "train": df.iloc[:train_end],
        "cali": df.iloc[train_end:cali_end],
        "test": df.iloc[cali_end:],
    }


# Data download
def fetch_series(
    series_id: str, api_key: str, start_date: str = BRENT_START_DATE
) -> pd.Series:
    """Download one FRED series as a date-indexed Series, dropping gaps."""
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date,
        "observation_end": END_DATE,
    }
    response = requests.get(FRED_BASE_URL, params=params, timeout=30)
    response.raise_for_status()

    observations = response.json()["observations"]
    df = pd.DataFrame(observations)

    # Parse dates and values; FRED uses "." for missing data.
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    series = df.set_index("date")["value"].dropna()
    series.name = series_id
    return series


# Processing
def build_dataset(api_key: str) -> pd.DataFrame:
    """Download Brent and CPI and compute the inflation-adjusted Brent price."""
    brent = fetch_series(BRENT_SERIES_ID, api_key)
    cpi = fetch_series(CPI_SERIES_ID, api_key, start_date=CPI_START_DATE)

    daily_index = brent.index
    df = pd.DataFrame(index=daily_index)
    df["Nominal_Brent"] = brent.reindex(daily_index)

    # Forward-fill the monthly CPI so each trading day inherits the most recent
    # monthly CPI reading.
    df["CPI"] = cpi.reindex(daily_index.union(cpi.index)).ffill().reindex(daily_index)

    # Drop any leading rows where CPI is still unknown.
    df = df.dropna(subset=["CPI"])

    # Inflation-adjusted (real) price expressed in latest-period dollars.
    latest_cpi = df["CPI"].iloc[-1]
    df["Real_Brent"] = df["Nominal_Brent"] * latest_cpi / df["CPI"]

    return df


def load_or_download() -> dict:
    """Load the cached datasets, downloading and caching them if absent."""
    os.makedirs(DATA_DIR, exist_ok=True)
    paths = {"brent": BRENT_PARQUET_PATH}

    if all(os.path.exists(p) for p in paths.values()):
        print("Loading cached datasets from Parquet")
        return {name: pd.read_parquet(p) for name, p in paths.items()}

    print("Cache not found. Downloading data from FRED...")
    df = build_dataset(get_api_key())
    datasets = {
        "brent": df[["Nominal_Brent", "CPI", "Real_Brent"]].dropna(
            subset=["Real_Brent"]
        )
    }
    for name, data in datasets.items():
        data.to_parquet(paths[name])
        print(f"Saved {name} dataset to {paths[name]}")
    return datasets


# Plotting
SERIES_COLOR = "#1a1a1a"
BAND_COLORS = {"train": "none", "cali": "#f2f2f2", "test": "#e4ecf3"}
LABELS = {"train": "Train", "cali": "Calibration", "test": "Test"}
EPISODE_COLOR = "#404040"

# Episodes shaded on the Brent figure
BRENT_EPISODES = [
    ("2008-07-03", "2009-02-28", "2008-07-03", "GFC", (0, 6), "center", "bottom"),
    ("2014-06-19", "2016-02-29", 0.94, "OPEC glut", (0, 0), "center", "bottom"),
    ("2020-03-06", "2020-06-30", "2020-04-21", "Covid", (-6, 0), "right", "bottom"),
    ("2022-02-24", "2022-06-30", 0.94, "Ukraine", (8, 0), "center", "bottom"),
    ("2026-02-28", "2026-07-01", "2026-04-07", "Iran", (2, 6), "right", "bottom"),
]


def _draw_segmented(ax, segments: dict, column: str) -> None:
    """Draw ``column`` with the train/calibration/test splits shaded."""
    full = pd.concat([seg[column] for seg in segments.values()])
    ax.plot(full.index, full.to_numpy(), color=SERIES_COLOR, linewidth=0.5)

    edges = {
        "train": (segments["train"].index[0], segments["cali"].index[0]),
        "cali": (segments["cali"].index[0], segments["test"].index[0]),
        "test": (segments["test"].index[0], segments["test"].index[-1]),
    }
    for name, (lo, hi) in edges.items():
        if BAND_COLORS[name] != "none":
            ax.axvspan(lo, hi, color=BAND_COLORS[name], linewidth=0, zorder=0)
        ax.annotate(
            LABELS[name],
            xy=(lo + (hi - lo) / 2, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#404040",
        )

    for boundary in (segments["cali"].index[0], segments["test"].index[0]):
        ax.axvline(boundary, color="#808080", linestyle=(0, (4, 2)), linewidth=0.5)

    # Pad by a few months so the year tick at each end is not clipped away.
    pad = pd.Timedelta(days=120)
    ax.set_xlim(full.index[0] - pad, full.index[-1] + pad)
    ax.grid(True, axis="y", zorder=0)
    ax.set_axisbelow(True)


PLOT_RC = {
    "font.family": "serif",
    "font.serif": ["cmr10", "CMU Serif", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
    "axes.formatter.use_mathtext": True,
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": "#cccccc",
    "grid.linewidth": 0.4,
    "lines.linewidth": 0.5,
    "savefig.dpi": 400,
    "pdf.fonttype": 42,
}

EPISODE_SHADE = "#efdcc7"
TAIL_COLOR = "#b2182b"


def _draw_episodes(ax, placement, y_frac=0.955, labels=None) -> None:
    """Shade the Brent episodes and label them inside the axes."""
    for start, stop, _anchor, label, _offset, _ha, _va in BRENT_EPISODES:
        if labels is not None and label not in labels:
            continue
        span = (pd.Timestamp(start), pd.Timestamp(stop))
        ax.axvspan(*span, color=EPISODE_SHADE, linewidth=0, zorder=0.5)
        if label not in placement:
            continue
        dx, ha = placement[label]
        ax.annotate(
            label,
            xy=(span[0] + (span[1] - span[0]) / 2, y_frac),
            xycoords=("data", "axes fraction"),
            xytext=(dx, 0),
            textcoords="offset points",
            ha=ha,
            va="top",
            fontsize=7,
            color=EPISODE_COLOR,
            zorder=4,
        )


def plot_brent_paper(values: pd.Series, proportions: list, filename: str) -> None:
    """Plot detrended real Brent from 2006, whole sample above, test below.

    The linear trend is fitted on train+calibration; connectors join the panels.
    """
    values = values.astype(float).dropna()
    residuals, _trend = _linear_detrend_pretest(values.to_numpy(), sum(proportions[:2]))
    df = pd.DataFrame({"Value": residuals}, index=values.index)
    segments = split_series(df, proportions)
    test = segments["test"]["Value"]
    band_end = int(sum(proportions[:2]) * len(residuals))
    q_lo, q_hi = (float(x) for x in np.quantile(residuals[:band_end], [0.10, 0.90]))

    with plt.rc_context(PLOT_RC):
        fig, (ax, ax_zoom) = plt.subplots(
            2,
            1,
            figsize=(5.6, 3.567),
            gridspec_kw={"height_ratios": [1.0, 1.1711], "hspace": 0.4119},
        )

        # whole sample
        _draw_segmented(ax, segments, "Value")
        ax.axhline(0.0, color="#808080", linestyle=(0, (2, 2)), linewidth=0.4, zorder=1)

        # Label placement is specific to this window: (horizontal offset in
        # points, alignment), keyed by episode.
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.15 * (hi - lo))
        _draw_episodes(
            ax,
            {
                "GFC": (0, "center"),
                "OPEC glut": (0, "center"),
                "Covid": (-3, "right"),
                "Ukraine": (-3, "right"),
                "Iran": (3, "right"),
            },
        )
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=7))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        # magnified test segment
        ax_zoom.set_facecolor(BAND_COLORS["test"])
        ax_zoom.plot(test.index, test.to_numpy(), color=SERIES_COLOR, linewidth=0.5)
        ax_zoom.axhline(
            0.0, color="#808080", linestyle=(0, (2, 2)), linewidth=0.4, zorder=1
        )
        pad = pd.Timedelta(days=20)
        ax_zoom.set_xlim(test.index[0] - pad, test.index[-1] + pad)
        zlo, zhi = float(test.min()), float(test.max())
        margin = 0.10 * (zhi - zlo)
        ax_zoom.set_ylim(zlo - margin, zhi + 2.2 * margin)
        ax_zoom.grid(True, axis="y", zorder=0)
        ax_zoom.set_axisbelow(True)
        _draw_episodes(ax_zoom, {"Iran": (-3, "right")}, labels={"Ukraine", "Iran"})
        # The Ukraine band is cut by the test boundary, so its midpoint lies off
        # this panel: anchor its label to the visible sliver instead.
        ax_zoom.annotate(
            "Ukraine",
            xy=(0.015, 0.90),
            xycoords="axes fraction",
            ha="left",
            va="top",
            fontsize=7,
            color=EPISODE_COLOR,
            zorder=4,
        )
        # Only the upper threshold falls inside the test panel: the series never
        # comes near the lower one over the test period.
        for level in (q_lo, q_hi):
            if ax_zoom.get_ylim()[0] < level < ax_zoom.get_ylim()[1]:
                ax_zoom.axhline(
                    level,
                    color=TAIL_COLOR,
                    linestyle=(0, (4, 2)),
                    linewidth=0.7,
                    zorder=3,
                )
        origins = test[(test > q_hi) | (test < q_lo)]
        ax_zoom.plot(
            origins.index,
            origins.to_numpy(),
            linestyle="none",
            marker="o",
            markersize=1.6,
            markerfacecolor=TAIL_COLOR,
            markeredgecolor="none",
            zorder=3,
        )
        ax_zoom.annotate(
            "tail threshold",
            xy=(0.30, q_hi),
            xycoords=("axes fraction", "data"),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6.5,
            color=TAIL_COLOR,
            zorder=4,
        )
        ax_zoom.xaxis.set_major_locator(mdates.YearLocator())
        ax_zoom.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        # connectors from the test band down to the magnified panel
        for x, corner in ((test.index[0], (0.0, 1.0)), (test.index[-1], (1.0, 1.0))):
            fig.add_artist(
                ConnectionPatch(
                    xyA=(mdates.date2num(x), ax.get_ylim()[0]),
                    coordsA=ax.transData,
                    xyB=corner,
                    coordsB=ax_zoom.transAxes,
                    color="#9aa5b1",
                    linestyle=(0, (3, 2)),
                    linewidth=0.5,
                )
            )

        # Two lines: the single-line label is longer than the axes are tall at
        # this aspect ratio and would be clipped.
        fig.supylabel("Real Brent (USD/barrel)", fontsize=9, x=0.028, ha="center")

        fig.subplots_adjust(left=0.125, right=0.99, bottom=0.0855, top=0.9273)

        stem = os.path.splitext(filename)[0]
        fig.savefig(os.path.join(PLOTS_DIR, f"{stem}.pdf"))
        plt.close(fig)
    print(f"Saved Brent plot to {PLOTS_DIR}/{stem}.pdf")


# Modeling: the main_simulations.py density pipeline, on the real series.

# data choice -> (parquet column holding the price level, human-readable title)
DATA_SERIES = {
    "brent": ("Real_Brent", "Inflation-adjusted Brent"),
}

# MDN mixture-component densities
DENSITY_CHOICES = [
    "gaussian",
    "student",
    "skewt",
    "skewnorm",
]

# Tail-weighting schemes for fit()
WEIGHTING_CHOICES = ["both", "sampler", "loss", "none"]

MODEL_CHOICES = [
    "mdn",
    "point_head",
    "kcde",
    "flexzboost",
    "flow",
    "conformal",
    "lstm",
    "xgboost",
]

APPLICATIONS_ROOT = os.path.join("outputs", "applications")


def _real_series(data_key: str, start_date: str | None = None) -> np.ndarray:
    """Load a real series as a 1-D float32 target array."""
    datasets = load_or_download()
    column, _ = DATA_SERIES[data_key]
    values = datasets[data_key][column].dropna()
    if start_date is not None:
        values = values.loc[values.index >= pd.Timestamp(start_date)]
    values = values.to_numpy(dtype=np.float32)
    return values


def _linear_detrend_pretest(series: np.ndarray, fit_fraction: float) -> tuple:
    """Remove a linear trend fit only through the train+calibration boundary."""
    fit_end = int(fit_fraction * len(series))
    time = np.arange(len(series), dtype=np.float64)
    slope, intercept = np.polyfit(time[:fit_end], series[:fit_end], deg=1)
    residuals = (series.astype(np.float64) - (intercept + slope * time)).astype(
        np.float32
    )
    return residuals, {
        "method": "linear",
        "fit_end_observation": fit_end,
        "intercept": float(intercept),
        "slope_per_observation": float(slope),
    }


def _inject_real_data(
    main,
    data_key: str,
    series: np.ndarray,
    horizon: int,
    density: str = "skewt",
    weighting: str = "both",
) -> None:
    """Point the main_simulations globals at the real series.

    `density` sets the MDN mixture-component head (default skewt, the headline
    model) and `weighting` the fit() tail-weighting scheme (default both). Both
    self-route their ablations to .../ablation/<density>/ and
    .../ablation/<weighting>/ (or .../ablation/<density>/<weighting>/) through
    main.backbone_dir(), exactly as on simulated data; both are ignored by the
    non-MDN models.
    """
    main.SERIES = series
    main.HORIZON = horizon
    main.PROPORTIONS = tuple(load_proportions())
    main.PROCESS = data_key
    main.DENSITY = density  # MDN head; skewt -> mdn/, else ablation/<density>/
    main.WEIGHTING = weighting  # both -> mdn/, else ablation/<weighting>/
    main.DGP = {"name": data_key, "kind": "real", "closed_form": False}
    main.APPLICATION_PINBALL_LEVELS = (0.05, 0.10, 0.90, 0.95)

    out_dir = Path(APPLICATIONS_ROOT) / data_key / f"horizon_{horizon}"
    out_dir.mkdir(parents=True, exist_ok=True)
    main.OUT_DIR = out_dir

    main._DATA_CACHE.clear()
    # Keep the single real realization fixed across any per-seed retrain path.
    main.build_series = lambda *args, **kwargs: main.SERIES


def _tail_band(main, series: np.ndarray) -> tuple:
    """Return the (lo, hi) bulk band.

    Test samples whose last observed lag falls outside it count as tail; see
    main_simulations._sample_region_mask.
    """
    p_train, p_val = main.PROPORTIONS[0], main.PROPORTIONS[1]
    test_slice = series[int((p_train + p_val) * len(series)) :]
    q05, q95 = np.quantile(test_slice, [0.10, 0.90])
    return float(q05), float(q95)


def _evaluate_density_single(
    main, model_label, out_dir, lags, q05, q95, recalibrate_fn
):
    """One train/validate -> test evaluation of a density model on real data."""
    out_dir.mkdir(parents=True, exist_ok=True)

    if main.RESCORE_ONLY and not main._has_test_density_cache(out_dir):
        main.logger.warning(
            f"--rescore: no {main._test_densities_path(out_dir)} -- nothing to "
            f"re-score, skipping {model_label}."
        )
        return None
    if main.RESCORE_ONLY:
        _, per_seed = main._rescore_from_cache(out_dir, q05, q95)
        metrics = per_seed[0]
        raw_frames = test_frames = cali_frames = None
    elif main.REPLAY_CACHED and main._has_density_cache(out_dir):
        _, per_seed, raw_frames, test_frames, cali_frames = main._replay_from_cache(
            out_dir, q05, q95
        )
        metrics = per_seed[0]
    else:
        data = main.get_data(lags)
        main.set_seed(main.SEED)
        grid_y = main._build_eval_grid(data, q05, q95)
        artifacts = {}
        recalibrated_density = recalibrate_fn(data, grid_y, artifacts)
        metrics = main._density_test_metrics(
            recalibrated_density, data, grid_y, q05, q95
        )
        raw_frames, test_frames, cali_frames = main._density_frames(
            main.SEED, data, grid_y, recalibrated_density, artifacts
        )
    if test_frames is not None:  # --rescore leaves the parquets untouched
        main._save_density_parquets(out_dir, raw_frames, test_frames, cali_frames)
    main._write_json(
        out_dir / "test_eval.json",
        {
            "model": model_label,
            "data": main.PROCESS,
            "horizon": main.HORIZON,
            "lags": lags,
            "seed": main.SEED,
            "metrics": metrics,
            "experiment": getattr(main, "EXPERIMENT_METADATA", None),
        },
    )
    agg = main._aggregate_over_seeds([metrics])
    main._print_eval_tables({model_label: agg}, [main.SEED], label_header="model")
    return metrics


def _evaluate_conformal_single(main, backbone, params, q05, q95, gammas):
    """Train the point predictor once and conformalize for each gamma."""
    lags = params["lags"]
    mdn, data = main._retrain_point_predictor_for_seed(backbone, params, main.SEED)
    y_test_np = data["y_test"].detach().cpu().numpy().astype(np.float64)
    inside_sample_mask = main._sample_region_mask(data, q05, q95)

    results = {}
    for g in gammas:
        regions = main._conformal_regions(mdn, data, aci_gamma=g)
        m = main._conformal_interval_metrics(
            regions, y_test_np, inside_sample_mask, alpha=main.CONFORMAL_ALPHA
        )
        d = main._conformal_gamma_dir(backbone, g)
        d.mkdir(parents=True, exist_ok=True)
        main._write_json(
            d / "test_eval.json",
            {
                "model": "conformal",
                "data": main.PROCESS,
                "horizon": main.HORIZON,
                "lags": lags,
                "seed": main.SEED,
                "aci_gamma": g,
                "metrics": m,
            },
        )
        results[main._gamma_tag(g)] = main._aggregate_over_seeds([m])
    main._print_eval_tables(results, [main.SEED], label_header="gamma")
    return results


# Point-forecast benchmarks (XGBoost, LSTM) -- evaluated with point metrics only.
def _np(tensor) -> np.ndarray:
    """Detach a torch tensor to a NumPy array."""
    return tensor.detach().cpu().numpy()


def _evaluate_point_single(main, model_label, out_dir, lags, pred, data, q05, q95):
    """Score a point forecast on the test set and persist it.

    MSE/MAE by bulk/tail region plus the no-change ratios. Shared by XGBoost and
    the LSTM.
    """
    y_test = _np(data["y_test"]).astype(np.float64)
    pred = np.asarray(pred, dtype=np.float64).ravel()
    metrics = main._point_region_metrics(pred, y_test, data, q05, q95)

    out_dir.mkdir(parents=True, exist_ok=True)
    main._save_point_parquet(
        out_dir,
        main._point_predictions_frame(main.SEED, y_test, main._np_last_lag(data), pred),
    )
    main._write_json(
        out_dir / "test_eval.json",
        {
            "model": model_label,
            "data": main.PROCESS,
            "horizon": main.HORIZON,
            "lags": lags,
            "seed": main.SEED,
            "metrics": metrics,
        },
    )

    fmt = main.format_number_4_digits
    main.logger.info(
        f"\n{'=' * 60}\nPOINT FORECAST TEST EVAL: {model_label}\n{'=' * 60}"
    )
    for key in ("mse", "mae", "mse_rel", "mae_rel"):
        r = metrics[key]
        main.logger.info(
            f"[{model_label}] {key:8s} | all={fmt(r['all'])} | "
            f"inside={fmt(r['inside'])} | outside={fmt(r['outside'])}"
        )
    main.logger.info(f"[{model_label}] point metrics saved to {out_dir}/test_eval.json")
    return metrics


def _rescore_point_single(main, model_label, out_dir, q05, q95):
    """Re-score a saved point forecast.

    None when there is no parquet, so the caller falls through to the fitted path.
    """
    if not main._has_point_cache(out_dir):
        main.logger.warning(
            f"--rescore: no {main._point_predictions_path(out_dir)} -- nothing to "
            f"re-score, skipping {model_label}."
        )
        return None
    _, per_seed = main._rescore_point_from_cache(out_dir, q05, q95)
    metrics = per_seed[0]
    main._write_json(
        out_dir / "test_eval.json",
        {
            "model": model_label,
            "data": main.PROCESS,
            "horizon": main.HORIZON,
            "seed": main.SEED,
            "metrics": metrics,
        },
    )
    fmt = main.format_number_4_digits
    for key in ("mse", "mae", "mse_rel", "mae_rel"):
        r = metrics[key]
        main.logger.info(
            f"[{model_label}] {key:8s} | all={fmt(r['all'])} | "
            f"inside={fmt(r['inside'])} | outside={fmt(r['outside'])}"
        )
    return metrics


def _build_xgb(main, n_estimators, max_depth, learning_rate):
    """Build a plain XGBoost regressor for direct point forecasting."""
    import xgboost as xgb

    return xgb.XGBRegressor(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        learning_rate=float(learning_rate),
        objective="reg:squarederror",
        tree_method="hist",
        random_state=main.SEED,
        n_jobs=main.n_process,
    )


def _run_xgboost_study(main, n_trials):
    """Tune XGBoost's point-forecast hyperparameters with Optuna.

    Same non-density search space as FlexZBoost, minimising calibration MSE.
    """
    out_dir = main.OUT_DIR / "xgboost"
    out_dir.mkdir(parents=True, exist_ok=True)

    def objective(trial):
        main.set_seed(main.SEED)
        lags = main.suggest_lags(trial)
        n_estimators = trial.suggest_int("n_estimators", 100, 2000, log=True)
        max_depth = trial.suggest_int("max_depth", 2, 10, step=2)
        learning_rate = trial.suggest_categorical(
            "learning_rate", main.FLEXZBOOST_LR_GRID
        )
        data = main.get_data(lags)
        model = _build_xgb(main, n_estimators, max_depth, learning_rate)
        model.fit(_np(data["X_train"]), _np(data["y_train"]))
        pred = model.predict(_np(data["X_cali"]))
        return float(np.mean((pred - _np(data["y_cali"])) ** 2))

    study = main.run_optuna_study(
        study_name=f"xgboost_{main.PROCESS}",
        label="xgboost (point forecaster)",
        progress_tag="[xgboost]",
        out_dir=out_dir,
        storage=f"sqlite:///{(out_dir / 'study.db').resolve()}",
        tracker_key="xgboost",
        objective=objective,
        n_trials=n_trials,
        timeout_minutes=None,
        seed=main.SEED,
    )

    best = dict(study.best_trial.params)
    main.logger.info(
        f"\n[xgboost] best val MSE: {study.best_trial.value:.5f} | params={best}"
    )
    main._write_json(
        out_dir / "best.json",
        {
            "model": "xgboost",
            "best_value_mse": study.best_trial.value,
            "params": best,
        },
    )
    main._write_yaml(
        out_dir / "config.yaml",
        {
            "model": "xgboost",
            "params": best,
            "fixed": {"horizon": main.HORIZON, "objective": "mse"},
        },
    )
    return best


def _load_xgboost_config(main):
    """Load the tuned XGBoost params from <xgboost>/best.json (for --evaluate)."""
    import json

    with open(main.OUT_DIR / "xgboost" / "best.json") as f:
        return json.load(f)["params"]


def _run_model(main, model, n_trials, evaluate, q05, q95):
    """Tune (unless --evaluate) then run a single test evaluation for one model."""
    backbone = "lstm"

    if model == "mdn":
        if not evaluate:
            main.run_study(backbone, n_trials, None)
        params = main.load_config_yaml(backbone)
        metrics = main.evaluate_backbone(backbone, params, q05, q95)
        agg = main._aggregate_over_seeds([metrics])
        main._print_eval_tables({"mdn": agg}, [main.SEED], label_header="model")

    elif model == "point_head":
        if main.RESCORE_ONLY:
            main.evaluate_point_head(backbone, q05, q95)
            return
        if not evaluate:
            if not main._config_path(backbone).exists():
                main.run_study(backbone, n_trials, None)
            density_params = main.load_config_yaml(backbone)
            main.run_point_head_study(backbone, density_params, n_trials, None)
        main.evaluate_point_head(backbone, q05, q95)

    elif model == "kcde":
        if not evaluate:
            main.run_kcde_study(n_trials, None)
        params = main.load_kcde_config()

        def recal(data, grid_y, artifacts):
            return main.recalibrate_test_kcde(
                params, data, grid_y, main.device, main.n_process, artifacts=artifacts
            )[1]

        _evaluate_density_single(
            main, "kcde", main._kcde_dir(), params["lags"], q05, q95, recal
        )

    elif model == "flexzboost":
        if not evaluate:
            main.run_flexzboost_study(n_trials, None)
        params = main.load_flexzboost_config()

        def recal(data, grid_y, artifacts):
            z_min, z_max = float(grid_y[0]), float(grid_y[-1])
            fz = main.build_and_fit_flexzboost(
                params, data, z_min, z_max, n_jobs=main.n_process, n_grid=main.N_GRID
            )
            return main.recalibrate_test_flexzboost(
                fz,
                data,
                grid_y,
                main.N_GRID,
                main.device,
                main.n_process,
                artifacts=artifacts,
            )[1]

        _evaluate_density_single(
            main, "flexzboost", main._flexzboost_dir(), params["lags"], q05, q95, recal
        )

    elif model == "flow":
        if not evaluate:
            main.run_flow_study(n_trials, None)
        params = main.load_flow_config()

        def recal(data, grid_y, artifacts):
            flow = main.build_flow(params, main.device)
            scalers = main.fit_flow_scalers(data)
            main.train_flow(
                flow,
                data,
                scalers,
                params["learning_rate"],
                main.device,
                max_epochs=main.MAX_EPOCHS,
                batch_size=main.BATCH_SIZE,
                max_norm=main.MAX_NORM,
                patience=main.PATIENCE,
                patience_scheduler=main.PATIENCE_SCHEDULER,
                factor_scheduler=main.FACTOR_SCHEDULER,
            )
            return main.recalibrate_test_flow(
                flow,
                scalers,
                data,
                grid_y,
                main.device,
                main.n_process,
                artifacts=artifacts,
            )[1]

        _evaluate_density_single(
            main, "flow", main._flow_dir(), params["lags"], q05, q95, recal
        )

    elif model == "conformal":
        if not evaluate:
            main.run_conformal_study(backbone, n_trials, None)
        params = main.load_conformal_config(backbone)[1]
        _evaluate_conformal_single(
            main, backbone, params, q05, q95, main.CONFORMAL_ACI_GAMMAS
        )

    elif model == "lstm":
        if main.RESCORE_ONLY:
            _rescore_point_single(main, "lstm", main.OUT_DIR / "lstm", q05, q95)
            return
        if not main._conformal_config_path(backbone).exists():
            raise FileNotFoundError(
                f"[lstm] needs the tuned conformal config at "
                f"{main._conformal_config_path(backbone)} -- lstm and conformal "
                f"share the same LSTM point predictor, so lstm reuses conformal's "
                f"best config rather than tuning its own. Run --model conformal "
                f"first (or --model all)."
            )
        params = main.load_conformal_config(backbone)[1]
        mdn, data = main._retrain_point_predictor_for_seed(backbone, params, main.SEED)
        pred = _np(mdn.pred_point(data["X_test"]))
        _evaluate_point_single(
            main, "lstm", main.OUT_DIR / "lstm", params["lags"], pred, data, q05, q95
        )

    elif model == "xgboost":
        if main.RESCORE_ONLY:
            _rescore_point_single(main, "xgboost", main.OUT_DIR / "xgboost", q05, q95)
            return
        # Point-forecast benchmark: a plain XGBoost regressor
        params = (
            _load_xgboost_config(main)
            if evaluate
            else _run_xgboost_study(main, n_trials)
        )
        lags = params["lags"]
        data = main.get_data(lags)
        main.set_seed(main.SEED)
        model_xgb = _build_xgb(
            main, params["n_estimators"], params["max_depth"], params["learning_rate"]
        )
        model_xgb.fit(_np(data["X_train"]), _np(data["y_train"]))
        pred = model_xgb.predict(_np(data["X_test"]))
        _evaluate_point_single(
            main, "xgboost", main.OUT_DIR / "xgboost", lags, pred, data, q05, q95
        )


def run_applications(
    data_key,
    models,
    horizon,
    n_trials,
    evaluate,
    density="skewt",
    weighting="both",
    force_refit=False,
    rescore=False,
    save_raw=True,
):
    """Run the main_simulations pipeline on a real FRED series."""
    import main_simulations as main

    is_detrended_brent = data_key == "brent"
    start_date = "2006-01-01" if is_detrended_brent else None
    series = _real_series(data_key, start_date=start_date)
    metadata = None
    q05 = q95 = None
    if is_detrended_brent:
        fit_fraction = sum(load_proportions()[:2])
        series, trend = _linear_detrend_pretest(series, fit_fraction)
        band_end = int(fit_fraction * len(series))
        q05, q95 = (float(x) for x in np.quantile(series[:band_end], [0.10, 0.90]))
        metadata = {
            "series": "Real_Brent",
            "start_date": start_date,
            "transformation": "linear_detrend_fit_train_plus_calibration_only",
            "tail_definition": "detrended_train_plus_calibration_q10_q90_on_last_lag",
            "tail_band": [q05, q95],
            "trend": trend,
        }
    _inject_real_data(main, data_key, series, horizon, density, weighting)
    # ``--evaluate`` ordinarily replays saved densities.  A forced refit still
    # loads the saved configuration (and therefore does not re-tune), but
    # rebuilds the model and PIT recalibrator from scratch.
    main.REPLAY_CACHED = evaluate and not force_refit
    main.RESCORE_ONLY = rescore
    main.FORCE_REFIT = force_refit
    main.SAVE_RAW = save_raw
    if q05 is None:
        q05, q95 = _tail_band(main, series)
    main.EXPERIMENT_METADATA = metadata
    main.logger.info(
        f"[applications] data={data_key} | N={len(series)} | horizon={horizon} | "
        f"density={density} | weighting={weighting} | "
        f"bulk/tail band [{q05:.4f}, {q95:.4f}] | "
        f"models={models} | seed={main.SEED}"
    )
    for model in models:
        _run_model(main, model, n_trials, evaluate, q05, q95)
        if metadata is not None and model in {"mdn", "kcde", "flexzboost", "flow"}:
            eval_paths = {
                "mdn": main.backbone_dir("lstm") / "test_eval.json",
                "kcde": main._kcde_dir() / "test_eval.json",
                "flexzboost": main._flexzboost_dir() / "test_eval.json",
                "flow": main._flow_dir() / "test_eval.json",
            }
            path = eval_paths[model]
            with open(path) as f:
                payload = json.load(f)
            payload["experiment"] = metadata
            main._write_json(path, payload)


# Entry point
def main() -> None:
    """Download or load the series and write the exploratory plots."""
    datasets = load_or_download()
    proportions = load_proportions()

    os.makedirs(PLOTS_DIR, exist_ok=True)

    plot_brent_paper(
        datasets["brent"]["Real_Brent"].loc["2006-01-01":],
        proportions,
        "brent_paper.png",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Density-forecast pipeline on real FRED "
        "series. With no --data, just (re)generates the level plots."
    )
    parser.add_argument(
        "--data",
        choices=list(DATA_SERIES),
        help="Which real series to model.",
    )
    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES + ["all"],
        default="all",
        help="Which model to tune + evaluate (default: all five).",
    )
    parser.add_argument(
        "--density",
        choices=DENSITY_CHOICES,
        default="skewt",
        help="MDN mixture-component density for --model mdn (default: skewt, the "
        "headline head at .../mdn/). Non-skewt heads run the ablation and write "
        "to .../ablation/<density>/. Ignored by the non-MDN models.",
    )
    parser.add_argument(
        "--weighting",
        choices=WEIGHTING_CHOICES,
        default="both",
        help="Tail-weighting scheme for fit() with --model mdn (default: both, "
        "the headline scheme at .../mdn/). Non-both schemes run the ablation and "
        "write to .../ablation/<weighting>/ (or .../ablation/<density>/<weighting>/ "
        "for a non-skewt head). Ignored by the non-MDN models.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="Forecast horizon h (predict Y_t from Y_{t-h}); default 1.",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=None,
        help="Optuna trials per model (required unless --evaluate is passed).",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Skip tuning and re-score from the saved raw_densities.parquet: only "
        "the recalibrator is refitted. Falls back to retraining from the saved "
        "config(s) when the parquets are not on disk yet.",
    )
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="Skip tuning, training AND recalibration: recompute the metrics "
        "straight from the saved test_densities.parquet. Use this to re-run a "
        "change to the METRICS, where --evaluate re-runs a change to the "
        "CALIBRATOR. Implies --evaluate.",
    )
    parser.add_argument(
        "--no_save_raw",
        dest="save_raw",
        action="store_false",
        help="Skip raw_densities.parquet (the UNcalibrated densities); they are "
        "saved by default -- see main_simulations.py --no_save_raw.",
    )
    parser.add_argument(
        "--refit",
        action="store_true",
        help="With --evaluate, use the saved configuration but force a fresh "
        "model fit and PIT recalibration instead of replaying saved Parquets.",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Just (re)generate the level plots and exit.",
    )
    args = parser.parse_args()

    if args.plots or args.data is None:
        main()
        raise SystemExit(0)

    if args.rescore:
        if args.refit:
            parser.error("--rescore and --refit are mutually exclusive")
        args.evaluate = True

    if not args.evaluate and args.n_trials is None:
        parser.error("--n_trials is required unless --evaluate is passed.")

    if args.refit and not args.evaluate:
        parser.error("--refit requires --evaluate")

    models = MODEL_CHOICES if args.model == "all" else [args.model]

    # point_head trains a regression head on a FROZEN skew-t backbone, so it only
    # supports the headline density -- mirror main_simulations.py's guard.
    if args.density != "skewt" and ("point_head" in models or args.model == "all"):
        parser.error(
            "--model point_head only supports --density skewt; run the density "
            "ablation with --model mdn."
        )

    run_applications(
        args.data,
        models,
        args.horizon,
        args.n_trials,
        args.evaluate,
        args.density,
        args.weighting,
        force_refit=args.refit,
        rescore=args.rescore,
        save_raw=args.save_raw,
    )
