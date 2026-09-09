"""Kernel conditional density baseline with independent bandwidths.

Build/fit/predict/recalibrate helpers only; the objective, config I/O and
multi-seed evaluation live in main_simulations.py.
"""

import numpy as np
from sklearn.preprocessing import RobustScaler

from src.forecast_methods.utils import recalibrate_test_density
from src.utils.model_constants import MODEL_CONSTANTS

_CFG = MODEL_CONSTANTS["kcde"]
KCDE_BANDWIDTH_GRID = list(_CFG["bandwidth_grid"])
KCDE_KERNELS = list(_CFG["kernels"])
KCDE_CHUNK_SIZE = _CFG["chunk_size"]


def kernel_eval(u, kernel):
    """Evaluate a univariate kernel at u."""
    u = np.asarray(u, dtype=np.float64)

    if kernel == "gaussian":
        return np.exp(-0.5 * u**2) / np.sqrt(2.0 * np.pi)

    if kernel == "epanechnikov":
        out = 0.75 * (1.0 - u**2)
        out[np.abs(u) > 1.0] = 0.0
        return out

    if kernel == "tricube":
        au = np.abs(u)
        out = (70.0 / 81.0) * (1.0 - au**3) ** 3
        out[au > 1.0] = 0.0
        return out

    raise ValueError(f"Unknown KCDE kernel: {kernel}")


def fit_kcde_scalers(data):
    """Fit RobustScaler objects on the training split only."""
    X_train = data["X_train"].detach().cpu().numpy().astype(np.float64)
    y_train = data["y_train"].detach().cpu().numpy().astype(np.float64).reshape(-1, 1)

    x_scaler = RobustScaler()
    y_scaler = RobustScaler()

    X_train_s = x_scaler.fit_transform(X_train)
    y_train_s = y_scaler.fit_transform(y_train).ravel()

    y_scale = float(y_scaler.scale_[0])
    y_scale = max(y_scale, 1e-12)

    return {
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "X_train_s": X_train_s,
        "y_train_s": y_train_s,
        "y_scale": y_scale,
    }


def kcde_grid(data, n_grid):
    """Build the tuning grid, matching the evaluation grid endpoints."""
    y_tr_np = data["y_train"].detach().cpu().numpy()
    return np.linspace(
        np.quantile(y_tr_np, 0.001), np.quantile(y_tr_np, 0.999), n_grid
    ).astype(np.float32)


def kcde_predict_density(
    data,
    X_query,
    grid_y,
    bandwidth_x,
    bandwidth_y,
    kernel,
    chunk_size=KCDE_CHUNK_SIZE,
):
    """Predict conditional densities p(y | x) on grid_y using direct KCDE.

    Two independent bandwidths are used: ``bandwidth_x`` scales the conditioning
    (lag) product kernel, ``bandwidth_y`` scales the response kernel. RobustScaler
    is fitted on the training split only. The returned density is converted back
    to the original y scale.
    """
    fitted = fit_kcde_scalers(data)

    X_train = fitted["X_train_s"]
    y_train = fitted["y_train_s"]
    x_scaler = fitted["x_scaler"]
    y_scaler = fitted["y_scaler"]
    y_scale = fitted["y_scale"]

    Xq = X_query.detach().cpu().numpy().astype(np.float64)
    Xq = x_scaler.transform(Xq)

    grid = np.asarray(grid_y, dtype=np.float64).reshape(-1, 1)
    grid_s = y_scaler.transform(grid).ravel()

    hx = float(bandwidth_x)
    hy = float(bandwidth_y)
    if hx <= 0 or hy <= 0:
        raise ValueError(f"KCDE bandwidths must be positive, got hx={hx}, hy={hy}")

    n_query = Xq.shape[0]
    n_grid = len(grid_s)
    out = np.empty((n_query, n_grid), dtype=np.float64)

    # Response (y) kernel -- bandwidth_y. Shape: (n_train, n_grid)
    Ky = kernel_eval((grid_s[None, :] - y_train[:, None]) / hy, kernel) / hy

    for start in range(0, n_query, chunk_size):
        stop = min(start + chunk_size, n_query)
        Xc = Xq[start:stop]

        # Conditioning (X) product kernel across lag dimensions -- bandwidth_x.
        # Shape before product: (chunk, n_train, n_features)
        Ux = (Xc[:, None, :] - X_train[None, :, :]) / hx
        Kx = kernel_eval(Ux, kernel).prod(axis=2)

        denom = Kx.sum(axis=1, keepdims=True)

        # Compact kernels can produce zero-neighbor rows in higher dimension.
        # Fallback: assign weight to the nearest training point.
        bad = denom[:, 0] <= 1e-14
        if np.any(bad):
            dist2 = np.sum(Ux[bad] ** 2, axis=2)
            nn = np.argmin(dist2, axis=1)

            local_bad_idx = np.where(bad)[0]
            Kx[local_bad_idx, :] = 0.0
            Kx[local_bad_idx, nn] = 1.0
            denom = Kx.sum(axis=1, keepdims=True)

        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            dens_s = (Kx @ Ky) / np.clip(denom, 1e-14, None)

        # Change of variables: density on original y scale.
        out[start:stop] = dens_s / y_scale

    return out


def kcde_val_loss(params, data, grid_y):
    """Return the validation CDE loss: negative log density at y_cali."""
    dens_cali = kcde_predict_density(
        data,
        data["X_cali"],
        grid_y,
        bandwidth_x=params["bandwidth_x"],
        bandwidth_y=params["bandwidth_y"],
        kernel=params["kernel"],
    )

    y_cali = data["y_cali"].detach().cpu().numpy().astype(np.float64)
    grid = np.asarray(grid_y, dtype=np.float64)

    dens_at_y = np.array(
        [
            np.interp(y_cali[i], grid, dens_cali[i], left=1e-12, right=1e-12)
            for i in range(len(y_cali))
        ],
        dtype=np.float64,
    )

    return float(-np.mean(np.log(np.clip(dens_at_y, 1e-12, None))))


def recalibrate_test_kcde(params, data, grid_y, device, n_jobs, artifacts=None):
    """Recalibrate KCDE predictive density using LocalPITRecalibrator."""
    return recalibrate_test_density(
        lambda X: kcde_predict_density(
            data,
            X,
            grid_y,
            bandwidth_x=params["bandwidth_x"],
            bandwidth_y=params["bandwidth_y"],
            kernel=params["kernel"],
        ),
        data,
        grid_y,
        device,
        n_jobs,
        artifacts=artifacts,
    )
