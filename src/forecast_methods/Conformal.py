"""Conformal-prediction baseline (MAPIE TimeSeriesRegressor: ACI).

A point-predictor backbone is tuned over the same search space as the MDN and
trained fully unweighted with fit_mse to minimize the validation MSE.

Build/fit + ACI-region helpers only; the objective, config/checkpoint I/O, and
multi-seed evaluation live in main_simulations.py.
"""

import numpy as np
import torch
from mapie.regression import TimeSeriesRegressor
from sklearn.base import BaseEstimator, RegressorMixin

from src.utils.model_constants import MODEL_CONSTANTS
from src.utils.setup_logger import setup_logger

_CFG = MODEL_CONSTANTS["conformal"]
CONFORMAL_ALPHA = _CFG["alpha"]  # 1 - confidence_level; naive-95% interval metrics
CONFORMAL_ACI_GAMMAS = list(_CFG["aci_gammas"])  # swept at eval time (0 => no adapt.)


class MDNPointRegressor(BaseEstimator, RegressorMixin):
    """Prefit sklearn wrapper exposing an MDN's tuned point head to MAPIE.

    The MDN is already trained (fit_mse); this only forwards .predict to
    mdn.pred_point on the original scale. Used exclusively with cv="prefit", so
    MAPIE never clones or refits the underlying torch model.
    """

    def __init__(self, mdn=None):
        """Wrap ``mdn`` in the estimator interface MAPIE expects."""
        self.mdn = mdn

    def fit(self, X=None, y=None):
        """Mark the wrapper fitted; the underlying model is already trained."""
        self.is_fitted_ = True
        self.fitted_ = True
        return self

    def predict(self, X):
        """Return the point predictions of the wrapped model."""
        Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        return self.mdn.pred_point(Xt).detach().cpu().numpy().astype(np.float64)

    def __sklearn_is_fitted__(self):
        """Report fitted to scikit-learn once a model is attached."""
        return self.mdn is not None


def unit_weights(n, device):
    """Return unit weights on the active device; training stays unweighted."""
    return torch.ones(n, device=device)


def fit_mse_unweighted(
    mdn,
    data,
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
    """Train backbone + point_head with fit_mse, fully UNWEIGHTED.

    weighting="none" already disables the sampler and forces unit loss weights;
    unit weight tensors are passed too so the call is unambiguous. Returns the
    best (unweighted) validation MSE.
    """
    return mdn.fit_mse(
        X_train=data["X_train"],
        y_train=data["y_train"],
        X_val=data["X_cali"],
        y_val=data["y_cali"],
        weights_train=unit_weights(len(data["X_train"]), device),
        weights_val=unit_weights(len(data["X_cali"]), device),
        max_epochs=max_epochs,
        learning_rate=lr,
        batch_size=batch_size,
        max_norm=max_norm,
        patience=patience,
        patience_scheduler=patience_scheduler,
        factor_scheduler=factor_scheduler,
        weighting="none",
    )


def conformal_regions(mdn, data, aci_gamma, alpha, seed, logger=None):
    """Build ACI prediction regions for the test set.

    The trained `mdn` (unweighted point predictor) is wrapped as a prefit sklearn
    regressor; MAPIE's TimeSeriesRegressor calibrates on the calibration split
    (cv="prefit", no refit) and predicts on the test set. ACI adapts alpha_t
    along the test stream via adapt_conformal_inference(gamma) (gamma=0 => static
    split-conformal); update()'s gamma is deprecated and must not be used here.

    Infinite / overly-wide intervals are capped to mu_hat +/- |eps|_max, where
    |eps|_max is the maximum absolute test residual of the point predictor.

    Returns a list where each regions[i] is a one-element [(lower, upper)],
    matching the format the interval-metric helpers expect.
    """
    logger = logger or setup_logger()

    cl = 1.0 - alpha
    X_ca = data["X_cali"].detach().cpu().numpy()
    y_ca = data["y_cali"].detach().cpu().numpy()
    X_te = data["X_test"].detach().cpu().numpy()
    y_te = data["y_test"].detach().cpu().numpy()

    est = MDNPointRegressor(mdn).fit()

    y_pred_test = np.ravel(est.predict(X_te))
    eps_max = float(np.max(np.abs(np.ravel(y_te) - y_pred_test)))

    # ACI -- adaptive, stepped over the test stream.
    m_aci = TimeSeriesRegressor(
        estimator=est, method="aci", cv="prefit", random_state=seed
    )
    m_aci.fit(X_ca, y_ca)
    regions = []
    n_capped = 0
    for i in range(len(X_te)):
        xi = X_te[i : i + 1]
        y_pred, p = m_aci.predict(xi, confidence_level=cl, allow_infinite_bounds=True)
        center = float(np.ravel(y_pred)[0])
        lo, hi = float(p[0, 0, 0]), float(p[0, 1, 0])
        # Cap each bound at mu_hat +/- |eps|_max (handles +/-inf and wide finite).
        new_lo = max(lo, center - eps_max)
        new_hi = min(hi, center + eps_max)
        if new_lo != lo or new_hi != hi:
            n_capped += 1
        regions.append([(new_lo, new_hi)])
        # ACI alpha_t recursion (the actual gamma-driven adaptation).
        m_aci.adapt_conformal_inference(xi, y_te[i : i + 1], gamma=aci_gamma)

    if n_capped:
        logger.info(
            f"[conformal] gamma={aci_gamma}: capped {n_capped}/{len(X_te)} "
            f"bound(s) at mu_hat +/- |eps|_max (|eps|_max={eps_max:.4g}, "
            f"max abs test residual)"
        )

    return regions
