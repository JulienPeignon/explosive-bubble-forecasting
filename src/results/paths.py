"""Path builders for the per-run artifacts.

Pure path arithmetic, apart from the directory ``algo_dir`` creates so the path
is usable immediately.
"""


def algo_dir(out_dir_root, name, subdir=None):
    """Return (creating) the artifact directory <root>/<name>[/<subdir>]."""
    d = out_dir_root / name if subdir is None else out_dir_root / name / subdir
    d.mkdir(parents=True, exist_ok=True)
    return d


def algo_study_db_url(algo_dir_path, filename="study.db"):
    """Return the absolute SQLite URL for a baseline's Optuna study."""
    return f"sqlite:///{algo_dir_path.resolve() / filename}"


def algo_config_path(algo_dir_path, filename="config.yaml"):
    """Return the config path inside an artifact directory."""
    return algo_dir_path / filename


def raw_densities_path(out_dir):
    """Return the path to <out_dir>/raw_densities.parquet."""
    return out_dir / "raw_densities.parquet"


def test_densities_path(out_dir):
    """Return the path to <out_dir>/test_densities.parquet."""
    return out_dir / "test_densities.parquet"


def point_predictions_path(out_dir):
    """Return the path to <out_dir>/point_predictions.parquet."""
    return out_dir / "point_predictions.parquet"


def pit_calibration_path(out_dir):
    """Return the path to <out_dir>/pit_calibration.parquet."""
    return out_dir / "pit_calibration.parquet"


def density_weights_path(out_dir):
    """Return the path to <out_dir>/density_model.pt."""
    return out_dir / "density_model.pt"


def point_head_weights_path(out_dir):
    """Return the path to <out_dir>/point_head_model.pt."""
    return out_dir / "point_head_model.pt"


def trials_path(out_dir):
    """Return the path to <out_dir>/trials.json."""
    return out_dir / "trials.json"


class ArtifactPaths:
    """Bundle the dir / study-db / config accessors around one directory callable.

    ``dir_fn`` may take positional args (backbone, gamma, ...); each accessor
    forwards them unchanged.
    """

    def __init__(self, dir_fn):
        """Bind the directory-resolving callable."""
        self.dir = dir_fn

    def study_db_url(self, *args, filename="study.db"):
        """Return the Optuna study URL for this artifact directory."""
        return algo_study_db_url(self.dir(*args), filename)

    def config_path(self, *args, filename="config.yaml"):
        """Return the config path for this artifact directory."""
        return algo_config_path(self.dir(*args), filename)
