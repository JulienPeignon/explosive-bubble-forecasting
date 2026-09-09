"""Shared helpers for the forecast methods: recalibration and tail weights."""

import numpy as np
import torch
from scipy import stats

from src.calibration.LocalPITRecalibrator import LocalPITRecalibrator
from src.calibration.metrics import probability_integral_transform
from src.utils.setup_logger import setup_logger

logger = setup_logger()

DEFAULT_NUM_BASIS = 3
SPLIT_MAX_LAGS = 10
ALPHAS = np.linspace(0.0, 1.0, 41)


def _as_numpy(x):
    """Float array from a numpy array or a torch tensor on any device."""
    if hasattr(x, "detach"):
        x = x.detach().cpu()
    return np.asarray(x, dtype=float)


def fit_and_apply_recalibrator(
    x_cali, pit_cali, x_test, raw_density, grid_y, n_jobs, num_basis=DEFAULT_NUM_BASIS
):
    """Fit a LocalPITRecalibrator on (x_cali, pit_cali) and apply it to `raw_density`.

    The single place the calibrator is configured, fitted and applied, so the
    live path and the cached re-evaluation path (which replays saved raw
    densities) can never drift apart.
    """
    calibrator = LocalPITRecalibrator(n_jobs=n_jobs, num_basis=num_basis)
    calibrator.fit(x_cali, pit_cali, alphas=ALPHAS)
    return calibrator.transform(x_test, raw_density, grid_y, verbose=False)


def recalibrate_test_density(
    density_fn,
    data,
    grid_y,
    device,
    n_jobs,
    num_basis=DEFAULT_NUM_BASIS,
    artifacts=None,
):
    """Recalibrate on the calibration split; return (raw, recalibrated) densities.

    Shared by every density method; ``density_fn`` (X -> (n, n_grid)) is the only
    part that differs between them.

    `artifacts`, when given, is filled with everything needed to refit the
    recalibrator later without rerunning the model: the calibration PITs and
    features it was fitted on, plus the raw test densities and features.
    """
    dens_cali = density_fn(data["X_cali"])
    cde_cali = torch.as_tensor(
        np.asarray(dens_cali), dtype=torch.float32, device=device
    )
    pit_cali = probability_integral_transform(cde_cali, grid_y, data["y_cali"])

    pred_density = np.asarray(density_fn(data["X_test"]))
    recalibrated_density = fit_and_apply_recalibrator(
        data["X_cali"],
        pit_cali,
        data["X_test"],
        pred_density,
        grid_y,
        n_jobs,
        num_basis=num_basis,
    )

    if artifacts is not None:
        artifacts["raw"] = pred_density
        artifacts["pit_cali"] = _as_numpy(pit_cali).ravel()
        artifacts["X_cali"] = _as_numpy(data["X_cali"])
        artifacts["X_test"] = _as_numpy(data["X_test"])
        artifacts["num_basis"] = int(num_basis)

    return pred_density, recalibrated_density


def generalized_boxplot(data, alpha=0.05, p=0.9):
    """Compute generalized-boxplot whiskers for skewed, heavy-tailed data.

    `data` is (n,) univariate or (n, p) multivariate; returns
    (lower_whisker, upper_whisker) as scalars (univariate) or (p,) arrays.

    The `alpha` tail budget is split two-sided over the fitted Tukey g-and-h:
    whiskers sit at the alpha/2 and 1-alpha/2 quantiles.
    """
    data = np.array(data)

    # Check if univariate or multivariate
    if data.ndim == 1:
        data = data.reshape(-1, 1)
        is_univariate = True
    else:
        is_univariate = False

    _, n_dims = data.shape

    lower_whiskers = np.zeros(n_dims)
    upper_whiskers = np.zeros(n_dims)

    # Compute whiskers for each dimension
    for dim in range(n_dims):
        data_dim = data[:, dim]

        # Standardize the data
        l0 = np.median(data_dim)
        s0 = np.percentile(data_dim, 75) - np.percentile(data_dim, 25)  # IQR

        if s0 < 1e-10:  # Handle constant data
            lower_whiskers[dim] = l0
            upper_whiskers[dim] = l0
            logger.warning(
                f"Dimension {dim}: constant data detected, using median as whiskers"
            )
            continue

        x_star = (data_dim - l0) / s0

        # Shift to obtain strictly positive values
        zeta = 0.1
        r = x_star - np.min(x_star) + zeta

        # Standardize to (0, 1)
        r_range = np.max(r) - np.min(r)
        if r_range < 1e-10:
            lower_whiskers[dim] = l0
            upper_whiskers[dim] = l0
            logger.warning(
                f"Dimension {dim}: insufficient range, using median as whiskers"
            )
            continue

        r_tilde = (r - np.min(r)) / r_range

        # Inverse normal transformation
        r_tilde = np.clip(r_tilde, 1e-10, 1 - 1e-10)
        w = stats.norm.ppf(r_tilde)

        # Standardize the w values
        w_median = np.median(w)
        w_iqr = np.percentile(w, 75) - np.percentile(w, 25)

        if w_iqr < 1e-10:
            lower_whiskers[dim] = l0
            upper_whiskers[dim] = l0
            logger.warning(
                f"Dimension {dim}: insufficient IQR in transformed space, "
                f"using median as whiskers"
            )
            continue

        w_star = (w - w_median) / (w_iqr / 1.3426)

        # Estimate g and h parameters
        zp = stats.norm.ppf(p)
        Qp = np.percentile(w_star, p * 100)
        Q1mp = np.percentile(w_star, (1 - p) * 100)

        # Estimate g
        if np.abs(Qp + Q1mp) > 1e-10:
            g = (1 / zp) * np.log(-Qp / Q1mp)
        else:
            g = 0

        # Estimate h
        if g != 0 and np.abs(Qp + Q1mp) > 1e-10:
            denom = Qp + Q1mp
            if np.abs(denom) > 1e-10:
                h = (2 * np.log(-g * Qp * Q1mp / denom)) / (zp**2)
            else:
                h = 0
        else:
            h = 0

        # Determine the standard-normal quantiles at which to place the whiskers.
        z_lower = stats.norm.ppf(alpha / 2)
        z_upper = stats.norm.ppf(1 - alpha / 2)

        def _whisker(z):
            """Map a normal quantile through the fitted g-and-h to data scale."""
            if np.abs(g) > 1e-10:
                xi = (1 / g) * (np.exp(g * z) - 1) * np.exp(h * z**2 / 2)
            else:
                xi = z * np.exp(h * z**2 / 2)
            transform = stats.norm.cdf(w_median + (w_iqr / 1.3426) * xi)
            return (transform * r_range + np.min(x_star)) * s0 + l0

        lower_whisker = _whisker(z_lower)
        upper_whisker = _whisker(z_upper)

        lower_whiskers[dim] = lower_whisker
        upper_whiskers[dim] = upper_whisker

        logger.info(
            f"Dimension {dim}: generalized boxplot boundaries: "
            f"{lower_whisker:.3f}, {upper_whisker:.3f}"
        )

    if is_univariate:
        return lower_whiskers[0], upper_whiskers[0]
    else:
        return lower_whiskers, upper_whiskers


def prepare_tensors(
    df,
    X,
    y,
    lags,
    horizon,
    proportions,
    device,
):
    """Prepare lagged train/val/test tensors with tail weighting."""
    logger = setup_logger()

    # Input validation
    if df is None and (X is None or y is None):
        raise ValueError("Either df OR both (X, y) must be provided")

    if df is not None and X is not None:
        logger.warning("Both df and (X, y) provided. Using X and y, ignoring df.")

    if len(proportions) != 3:
        raise ValueError("proportions must be a 3-tuple (train, val, test)")
    if any(p < 0 for p in proportions):
        raise ValueError("proportions must be non-negative")
    if abs(sum(proportions) - 1.0) > 1e-8:
        raise ValueError(f"proportions must sum to 1, got {sum(proportions)}")

    p_train, p_val, p_test = proportions

    # Resolve data
    if X is not None and y is not None:
        X_data = X.flatten() if isinstance(X, np.ndarray) else np.array(X).flatten()
        y_data = y.flatten() if isinstance(y, np.ndarray) else np.array(y).flatten()
        logger.info(f"Using provided X and y arrays: X{X_data.shape}, y{y_data.shape}")
    else:
        if df.shape[1] > 1:
            logger.warning(
                f"DataFrame has {df.shape[1]} columns, using only the first "
                f"column for univariate mode"
            )
        X_data = df.iloc[:, 0].values if hasattr(df, "iloc") else df.values.flatten()
        y_data = X_data

    # Build lagged samples
    X_list, y_list = [], []
    max_t = min(len(X_data), len(y_data) - horizon + 1)
    for t in range(lags, max_t):
        X_list.append(X_data[t - lags : t])
        y_list.append(y_data[t + horizon - 1])

    X_array = np.array(X_list, dtype=np.float32)  # (n_samples, lags)
    y_array = np.array(y_list, dtype=np.float32)  # (n_samples,)

    X_tensor = torch.tensor(X_array, device=device)
    y_tensor = torch.tensor(y_array, device=device)
    n_samples = len(X_tensor)

    if lags > SPLIT_MAX_LAGS:
        raise ValueError(
            f"lags={lags} exceeds SPLIT_MAX_LAGS={SPLIT_MAX_LAGS}; raise the "
            f"constant (and re-run every method) so all splits stay aligned."
        )
    gap = SPLIT_MAX_LAGS + horizon - 1

    n_ref = max_t  # lag-independent: min(len(X_data), len(y_data) - horizon + 1)
    t = np.arange(n_samples) + lags  # series clock for each sample
    t_train_end = int(p_train * n_ref)
    t_val_end = t_train_end + int(p_val * n_ref)

    train_sel = t < t_train_end
    if not train_sel.any():
        raise ValueError(
            f"Empty training set (n_samples={n_samples}, proportions={proportions}, "
            f"gap={gap}). Provide more data or adjust proportions."
        )
    X_train, y_train = X_tensor[train_sel], y_tensor[train_sel]

    # Validation partition
    if p_val > 0:
        hi = t_val_end if p_test > 0 else n_ref
        val_sel = (t >= t_train_end + gap) & (t < hi)
        if not val_sel.any():
            raise ValueError(
                f"Empty/invalid validation set: t in [{t_train_end + gap}, {hi}), "
                f"n_ref={n_ref}, gap={gap}. Adjust proportions/lags/horizon."
            )
        X_val, y_val = X_tensor[val_sel], y_tensor[val_sel]
    else:
        t_val_end = t_train_end
        X_val, y_val = None, None

    # Test partition
    if p_test > 0:
        test_sel = t >= t_val_end + gap
        if not test_sel.any():
            raise ValueError(
                f"Empty test set: test starts at t={t_val_end + gap} >= n_ref={n_ref}, "
                f"gap={gap}. Adjust proportions/lags/horizon."
            )
        X_test, y_test = X_tensor[test_sel], y_tensor[test_sel]
    else:
        X_test, y_test = None, None

    logger.info(
        f"Split (samples): train={len(X_train)}, "
        f"val={0 if X_val is None else len(X_val)}, "
        f"test={0 if X_test is None else len(X_test)} | "
        f"gap={gap} (SPLIT_MAX_LAGS={SPLIT_MAX_LAGS}, lags={lags}, horizon={horizon})"
    )

    # Tail boundaries.
    train_series_np = np.asarray(X_data[:t_train_end], dtype=np.float64)
    lower_bound, upper_bound = generalized_boxplot(train_series_np)
    logger.info(
        f"Tail bounds estimated on the TRAIN series only "
        f"({len(train_series_np)} observations): "
        f"lower={lower_bound:.3f}, upper={upper_bound:.3f}"
    )

    # Weights.
    weights_train = torch.ones(len(X_train), device=device)
    last_train = X_train[:, -1]
    is_tail = (last_train < lower_bound) | (last_train > upper_bound)
    tail_count = is_tail.sum().item()
    logger.info(f"Tail samples (train) = {tail_count}, {tail_count / len(X_train):.2%}")
    if tail_count > 0:
        inverse_tail_proportion = np.sqrt(len(X_train) / tail_count)
        weights_train[is_tail] = inverse_tail_proportion
        # Pre-normalize to mean 1
        weight_norm = weights_train.mean().item()
        weights_train /= weight_norm
        logger.info(
            f"Tail weight = {inverse_tail_proportion:.3f} "
            f"(mean-1 normalized: tail={inverse_tail_proportion / weight_norm:.3f}, "
            f"bulk={1 / weight_norm:.3f})"
        )
    else:
        inverse_tail_proportion = None
        weight_norm = 1.0
    weights_train = weights_train.cpu()

    if X_val is not None:
        weights_val = torch.ones(len(X_val), device=device)
        last_val = X_val[:, -1]
        is_tail_val = (last_val < lower_bound) | (last_val > upper_bound)
        if inverse_tail_proportion is not None:
            weights_val[is_tail_val] = inverse_tail_proportion
            weights_val /= weight_norm
        weights_val = weights_val.cpu()
    else:
        weights_val = None

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        weights_train,
        weights_val,
    )
