"""Load configs/model_constants.yaml once and expose it as MODEL_CONSTANTS."""

from pathlib import Path

import yaml

_MODEL_CONSTANTS_PATH = Path("configs") / "model_constants.yaml"

with open(_MODEL_CONSTANTS_PATH) as _f:
    MODEL_CONSTANTS = yaml.safe_load(_f)
