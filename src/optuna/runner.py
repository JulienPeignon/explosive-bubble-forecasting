"""Generic Optuna study driver shared by every model's ``run_*_study``.

Factors out the identical create-study / optimize / trials-tracker / logging
boilerplate so each model only supplies its objective plus the model-specific
"finalize best trial" step (config YAML, checkpoint retrain).
"""

import optuna

from src.results.io import load_trials, save_trials
from src.utils.setup_logger import setup_logger

logger = setup_logger()


def run_optuna_study(
    *,
    study_name,
    label,
    progress_tag,
    out_dir,
    storage,
    tracker_key,
    objective,
    n_trials,
    timeout_minutes,
    seed,
    header_prefix="OPTIMIZING",
    header_sep="  ",
    direction="minimize",
):
    """Run (or resume) an Optuna study and return it.

    Handles the shared lifecycle: read the on-disk trials tracker, log the
    banner, create/resume the study (TPESampler seeded with ``seed``, persisted
    to ``storage``), optimize until ``n_trials`` or ``timeout_minutes`` is hit,
    then log + persist the new trial count. The caller is responsible for
    reading ``study.best_trial`` and writing its own artifacts.

    ``label`` / ``progress_tag`` / ``header_prefix`` reproduce each model's exact
    log strings; ``tracker_key`` is the per-artifact key in the trials tracker.
    """
    tracker = load_trials(out_dir)
    trials_before = tracker.get(tracker_key, 0)
    logger.info(
        f"\n{'=' * 60}\n{header_prefix}: {label}{header_sep}"
        f"(trials already done: {trials_before})\n{'=' * 60}"
    )

    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=seed),
        storage=storage,
        load_if_exists=True,  # always resume from the on-disk study
    )
    timeout = None if timeout_minutes is None else timeout_minutes * 60
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        gc_after_trial=True,
    )

    total_trials = len(study.trials)
    logger.info(
        f"{progress_tag} {total_trials - trials_before} new trial(s) this run | "
        f"{total_trials} total"
    )
    tracker[tracker_key] = total_trials
    save_trials(out_dir, tracker)

    return study
