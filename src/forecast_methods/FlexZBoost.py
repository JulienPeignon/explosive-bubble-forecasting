"""FlexZBoost baseline: FlexCode density estimation with an XGBoost regressor.

Build/fit/predict/recalibrate helpers only; the objective, config I/O and
multi-seed evaluation live in main_simulations.py.
"""

import flexcode
import numpy as np
from flexcode.regression_models import XGBoost

from src.forecast_methods.utils import recalibrate_test_density
from src.utils.model_constants import MODEL_CONSTANTS

_CFG = MODEL_CONSTANTS["flexzboost"]
FLEXZBOOST_MAX_BASIS = _CFG["max_basis"]

FLEXZBOOST_BUMP_GRID = np.linspace(
    _CFG["bump_grid"]["start"], _CFG["bump_grid"]["stop"], _CFG["bump_grid"]["num"]
)
FLEXZBOOST_SHARPEN_GRID = np.linspace(
    _CFG["sharpen_grid"]["start"],
    _CFG["sharpen_grid"]["stop"],
    _CFG["sharpen_grid"]["num"],
)
FLEXZBOOST_LR_GRID = list(_CFG["lr_grid"])


def flexzboost_grid_bounds(data):
    """Return (z_min, z_max) for FlexCode: the train [0.1%, 99.9%] quantiles."""
    y_tr = data["y_train"].detach().cpu().numpy()
    return float(np.quantile(y_tr, 0.001)), float(np.quantile(y_tr, 0.999))


def build_and_fit_flexzboost(params, data, z_min, z_max, n_jobs, n_grid):
    """Build, fit, and tune a FlexCode(XGBoost) model; return the tuned model.

    .tune() runs on the calibration set (best-basis truncation + optional
    bump/sharpen).
    """

    class _XGBoostWithJobs(XGBoost):
        def __init__(self, max_basis, regression_params, *args, **kwargs):
            kwargs["n_jobs"] = n_jobs
            super().__init__(max_basis, regression_params, *args, **kwargs)

    fz = flexcode.FlexCodeModel(
        _XGBoostWithJobs,
        max_basis=params["max_basis"],
        basis_system=params["basis_system"],
        z_min=z_min,
        z_max=z_max,
        regression_params={
            "verbosity": 0,
            "n_jobs": n_jobs,
            "objective": "reg:squarederror",
            "n_estimators": params.get("n_estimators", 100),
            "max_depth": params.get("max_depth", 6),
            "learning_rate": params.get("learning_rate", 0.3),
        },
    )
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        fz.fit(
            x_train=data["X_train"].detach().cpu().numpy(),
            z_train=data["y_train"].detach().cpu().numpy(),
        )
        # Validation-driven model selection on the calibration set
        fz.tune(
            data["X_cali"].detach().cpu().numpy(),
            data["y_cali"].detach().cpu().numpy(),
            bump_threshold_grid=FLEXZBOOST_BUMP_GRID,
            sharpen_grid=FLEXZBOOST_SHARPEN_GRID,
            n_grid=n_grid,
        )
    return fz


def flexzboost_density(fz, X, n_grid):
    """Predictive density of a fitted FlexCode model on the shared grid."""
    X_np = X.detach().cpu().numpy() if hasattr(X, "detach") else np.asarray(X)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        dens, _ = fz.predict(X_np, n_grid=n_grid)
    return np.asarray(dens, dtype=np.float64)


def recalibrate_test_flexzboost(
    fz, data, grid_y, n_grid, device, n_jobs, artifacts=None
):
    """Recalibrate the FlexCode predictive density on the test set."""
    return recalibrate_test_density(
        lambda X: flexzboost_density(fz, X, n_grid),
        data,
        grid_y,
        device,
        n_jobs,
        artifacts=artifacts,
    )
