"""Helpers operating directly on Optuna Study/Trial objects."""


def resolved_params(study):
    """Return the best trial's params as a plain dict."""
    return dict(study.best_trial.params)
