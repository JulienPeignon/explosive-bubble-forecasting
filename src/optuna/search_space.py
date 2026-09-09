"""Shared search-space building blocks for the Optuna objectives.

Grids and ranges are read from configs/model_constants.yaml so they are
defined once instead of hardcoded per objective.
"""

from src.utils.model_constants import MODEL_CONSTANTS

_CFG = MODEL_CONSTANTS["search_space"]
LR_GRID = list(_CFG["lr_grid"])
WIDTH_GRID = list(_CFG["width_grid"])
DEPTH_RANGE = tuple(_CFG["depth_range"])
LAGS_RANGE = (1, 10)


def suggest_lags(trial):
    """Suggest the number of conditioning lags."""
    low, high = LAGS_RANGE
    return trial.suggest_int("lags", low, high)


def suggest_learning_rate(trial):
    """Suggest the learning rate from the configured grid."""
    return trial.suggest_categorical("learning_rate", LR_GRID)


def suggest_dropout(trial):
    """Suggest the dropout rate."""
    return trial.suggest_float("dropout", 0.0, 0.2, step=0.05)


def suggest_backbone_arch(trial, backbone, s):
    """Append the backbone-specific depth/width params to `s` in place."""
    if backbone != "lstm":
        raise ValueError(f"Unknown backbone: {backbone}")
    s["lstm_depth"] = trial.suggest_int("lstm_depth", *DEPTH_RANGE)
    s["lstm_width"] = trial.suggest_categorical("lstm_width", WIDTH_GRID)
    return s
