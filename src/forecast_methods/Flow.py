"""Conditional normalizing-flow baseline (zuko rational-quadratic neural spline flow).

An unweighted conditional density estimator p(y | lags), trained by maximum
likelihood to minimize the validation NLL under the shared fixed protocol. NSF
is expressive within a single dimension via monotone rational-quadratic splines
(a RealNVP/MAF flow would degenerate to a conditional Gaussian on a scalar
target). Inputs/targets are RobustScaled on the train split; the density is
mapped back to original y units via p_Y(y) = p_Z(z) / y_scale.

Build/fit/predict/recalibrate helpers only; the objective, config I/O, and
multi-seed evaluation live in main_simulations.py.
"""

import numpy as np
import torch
import zuko
from sklearn.preprocessing import RobustScaler

from src.forecast_methods.utils import recalibrate_test_density
from src.optuna.search_space import DEPTH_RANGE, LR_GRID, WIDTH_GRID
from src.utils.model_constants import MODEL_CONSTANTS

_CFG = MODEL_CONSTANTS["flow"]
FLOW_LR_GRID = LR_GRID
FLOW_WIDTH_GRID = WIDTH_GRID
FLOW_BINS_GRID = list(_CFG["bins_grid"])
FLOW_TRANSFORMS_RANGE = tuple(_CFG["transforms_range"])  # density-resolution axis
# (mirrors n_mixtures); numerically the same range as DEPTH_RANGE today but a
# distinct search axis, so it is NOT tied to DEPTH_RANGE / search_space.yaml.
FLOW_DEPTH_RANGE = DEPTH_RANGE
FLOW_CHUNK_SIZE = _CFG["chunk_size"]


def build_flow(params, device):
    """Build a conditional NSF (rational-quadratic neural spline flow).

    features=1 (scalar target), context=lags (the conditioning vector). The
    conditioner is an MLP of `hidden_depth` layers each `hidden_width` wide.
    """
    hidden_features = [int(params["hidden_width"])] * int(params["hidden_depth"])
    flow = zuko.flows.NSF(
        features=1,
        context=int(params["lags"]),
        transforms=int(params["transforms"]),
        bins=int(params["bins"]),
        hidden_features=hidden_features,
    )
    return flow.to(device)


def fit_flow_scalers(data):
    """Fit RobustScaler objects on the TRAINING split only (X and y).

    Returns the scalers plus y_scale (the IQR), which sets the change-of-
    variables Jacobian back to the original target scale.
    """
    X_train = data["X_train"].detach().cpu().numpy().astype(np.float64)
    y_train = data["y_train"].detach().cpu().numpy().astype(np.float64).reshape(-1, 1)

    x_scaler = RobustScaler().fit(X_train)
    y_scaler = RobustScaler().fit(y_train)

    y_scale = max(float(y_scaler.scale_[0]), 1e-12)
    return {"x_scaler": x_scaler, "y_scaler": y_scaler, "y_scale": y_scale}


def flow_scale_X(scalers, X, device):
    """RobustScale a context tensor and return it as float32 on `device`."""
    X_np = X.detach().cpu().numpy().astype(np.float64)
    X_s = scalers["x_scaler"].transform(X_np)
    return torch.as_tensor(X_s, dtype=torch.float32, device=device)


def flow_scale_y(scalers, y, device):
    """RobustScale a target tensor to a (n, 1) float32 tensor on `device`."""
    y_np = y.detach().cpu().numpy().astype(np.float64).reshape(-1, 1)
    y_s = scalers["y_scaler"].transform(y_np)
    return torch.as_tensor(y_s, dtype=torch.float32, device=device)


def flow_val_nll(flow, scalers, data, device):
    """Return the calibration-split NLL, in original y units.

    log p_Y(y) = log p_Z(z) - log(y_scale), so the original-space NLL is the
    scaled NLL plus log(y_scale). Computing it in original units makes the value
    directly comparable across trials (whose y_scale differs with lags) and with
    the MDN / KCDE NLLs.
    """
    flow.eval()
    Xca = flow_scale_X(scalers, data["X_cali"], device)
    yca = flow_scale_y(scalers, data["y_cali"], device)
    with torch.no_grad():
        log_pz = flow(Xca).log_prob(yca)  # (n,)
    nll_scaled = float((-log_pz).mean().item())
    return nll_scaled + float(np.log(scalers["y_scale"]))


def train_flow(
    flow,
    data,
    scalers,
    lr,
    device,
    *,
    max_epochs,
    batch_size,
    max_norm,
    patience,
    patience_scheduler,
    factor_scheduler,
):
    """Train the flow by maximum likelihood on the shared fixed protocol.

    Adam + ReduceLROnPlateau, gradient clipping at ``max_norm``, early stopping
    on the calibration-set NLL. Unweighted. Restores the best weights in place
    and returns the best validation NLL in original y units.
    """
    Xtr = flow_scale_X(scalers, data["X_train"], device)
    ytr = flow_scale_y(scalers, data["y_train"], device)
    n = Xtr.shape[0]

    optimizer = torch.optim.Adam(flow.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=factor_scheduler, patience=patience_scheduler
    )

    best_nll = float("inf")
    best_state = None
    epochs_no_improve = 0

    for _epoch in range(max_epochs):
        flow.train()
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            optimizer.zero_grad()
            loss = -flow(Xtr[idx]).log_prob(ytr[idx]).mean()
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm)
            optimizer.step()

        val_nll = flow_val_nll(flow, scalers, data, device)
        scheduler.step(val_nll)

        if np.isfinite(val_nll) and val_nll < best_nll - 1e-6:
            best_nll = val_nll
            best_state = {k: v.detach().clone() for k, v in flow.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_state is not None:
        flow.load_state_dict(best_state)
    return best_nll


def flow_density(flow, scalers, X, grid_y, device, chunk_size=FLOW_CHUNK_SIZE):
    """Predictive density p(y | x) on the shared grid, in original y units.

    dens[i, j] = p(grid_y[j] | X[i]). The flow is evaluated in scaled space and
    converted back via the change of variables (/ y_scale).
    """
    flow.eval()
    Xq = flow_scale_X(scalers, X, device)  # (n, lags) on device
    grid = np.asarray(grid_y, dtype=np.float64).reshape(-1, 1)
    grid_s = scalers["y_scaler"].transform(grid).ravel()
    grid_t = torch.as_tensor(grid_s, dtype=torch.float32, device=device)  # (G,)
    G = grid_t.shape[0]

    n_query = Xq.shape[0]
    out = np.empty((n_query, G), dtype=np.float64)

    with torch.no_grad():
        for start in range(0, n_query, chunk_size):
            stop = min(start + chunk_size, n_query)
            Xc = Xq[start:stop]  # (c, lags)
            c = Xc.shape[0]
            dist = flow(Xc)  # batch_shape (c,)
            # grid broadcast to (G, c, 1) -> log_prob (G, c)
            gy = grid_t.reshape(G, 1, 1).expand(G, c, 1)
            log_pz = dist.log_prob(gy)  # (G, c)
            dens_s = log_pz.exp().T  # (c, G), scaled space
            out[start:stop] = dens_s.detach().cpu().numpy() / scalers["y_scale"]

    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def recalibrate_test_flow(flow, scalers, data, grid_y, device, n_jobs, artifacts=None):
    """Recalibrate the flow predictive density on the calibration split."""
    return recalibrate_test_density(
        lambda X: flow_density(flow, scalers, X, grid_y, device),
        data,
        grid_y,
        device,
        n_jobs,
        artifacts=artifacts,
    )
