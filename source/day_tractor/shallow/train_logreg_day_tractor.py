"""
Logistic Regression activity classifier for the day-tractor dataset.

Training data is loaded from the three pre-split Excel files produced by
``split_dataset.py``.  A random hyper-parameter search is performed over
``N_TRIALS`` configurations; results are appended to a shared summary
spreadsheet for cross-trial comparison.

Sliding windows from :class:`TractorActivityDataset` are flattened to
``(n_samples, window_size * n_features)`` before being passed to sklearn.
``window_size`` is a hyper-parameter, so datasets are re-created each trial.

The ``saga`` solver is fixed because it is the only sklearn solver that
supports all four penalty types (``l1``, ``l2``, ``elasticnet``, ``None``).
``l1_ratio`` is always passed to the model; it is silently ignored by sklearn
when ``penalty`` is not ``'elasticnet'``.
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import loguniform, randint
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from day_tractor.data.dataset import TractorActivityDataset
from shared_modules.logging_setup import setup_logging
from shared_modules.timing import timeit

logger = logging.getLogger(__name__)

# ── paths ─────────────────────────────────────────────────────────────────────
DATA_DIR    = Path(__file__).resolve().parents[3] / "data" / "day_tractor"
RESULTS_DIR = Path(__file__).resolve().parents[3] / "results" / "day_tractor" / "shallow" / "LR"

# ── random search configuration ───────────────────────────────────────────────
N_TRIALS: int = 30
SEARCH_SEED: int | None = 0

HP_DISTRIBUTIONS: dict[str, Any] = {
    "window_size": randint(10, 51),
    "penalty":     ["l1", "l2", "elasticnet", None],
    "C":           loguniform(1e-2, 1e2),
    "l1_ratio":    [0.1, 0.3, 0.5, 0.7, 0.9],
}

HP_FIXED: dict[str, Any] = {
    "seed":              42,
    "use_class_weights": True,
    "solver":            "saga",
    "max_iter":          1000,
}
# ─────────────────────────────────────────────────────────────────────────────


def _extract_arrays(ds: TractorActivityDataset) -> tuple[np.ndarray, np.ndarray]:
    """Flatten windows to 2-D and return labels.

    Args:
        ds: Loaded dataset instance.

    Returns:
        Tuple ``(X, y)`` where ``X`` has shape
        ``(n_windows, window_size * n_features)`` and ``y`` has shape
        ``(n_windows,)``.
    """
    X = ds.windows.reshape(len(ds.windows), -1)
    y = ds.labels
    return X, y


def _align_proba(proba: np.ndarray, model_classes: np.ndarray, n_classes: int) -> np.ndarray:
    """Expand ``predict_proba`` output to cover all encoder classes.

    ``LogisticRegression.predict_proba`` only includes columns for classes
    that appeared in the training set.  If a class had zero training windows its
    column is absent, which would cause shape mismatches in the metric
    functions.  This function inserts zero-probability columns for any missing
    class.

    Args:
        proba: Raw output of ``predict_proba``, shape ``(n, len(model_classes))``.
        model_classes: Integer class indices known to the fitted model
            (``model.classes_``).
        n_classes: Total number of classes in the label encoder.

    Returns:
        Array of shape ``(n, n_classes)`` with zeros for missing classes.
    """
    if proba.shape[1] == n_classes:
        return proba
    full = np.zeros((proba.shape[0], n_classes), dtype=np.float64)
    full[:, model_classes] = proba
    return full


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
) -> dict[str, float]:
    """Return accuracy, weighted and macro precision / recall / F1, and weighted AUC.

    Args:
        y_true: Ground-truth class indices.
        y_pred: Predicted class indices.
        y_probs: Class probabilities, shape ``(n, n_classes)``.

    Returns:
        Dictionary of metric name → scalar value.
    """
    all_labels = np.arange(y_probs.shape[1])
    try:
        auc = float(roc_auc_score(
            y_true, y_probs, multi_class="ovr", average="weighted", labels=all_labels
        ))
    except ValueError:
        auc = float("nan")

    return {
        "accuracy":        float((y_pred == y_true).mean()),
        "precision":       float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall":          float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1":              float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "auc":             auc,
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall":    float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1":        float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def write_predictions(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dir_name: str,
    subset_type: str,
) -> None:
    """Write an Excel file with flattened inputs and predictions for one split.

    Args:
        X: Flattened feature array, shape ``(n, window_size * n_features)``.
        y_true: Ground-truth class indices.
        y_pred: Predicted class indices.
        dir_name: Directory where the file is written.
        subset_type: Label used in the filename (``"train"``, ``"val"``, ``"test"``).
    """
    df_out = pd.DataFrame(X)
    df_out["predicted"]     = y_pred
    df_out["y_groundtruth"] = y_true
    df_out.to_excel(os.path.join(dir_name, f"{subset_type}_predictions.xlsx"), index=False)


@timeit
def play(
    verbose: bool = True,
    save_eps: bool = True,
    **hyperparameters: Any,
) -> LogisticRegression:
    """Fit and evaluate a Logistic Regression classifier on the day-tractor splits.

    Loads the three pre-split Excel files, builds datasets with the trial's
    ``window_size``, fits the model, and writes metrics, plots, and predictions
    to a timestamped results sub-directory.

    Args:
        verbose: When ``True``, log dataset statistics.
        save_eps: When ``True``, also save EPS versions of all figures.
        **hyperparameters: Training configuration.  Expected keys:
            ``window_size``, ``penalty``, ``C``, ``l1_ratio``,
            ``solver``, ``max_iter``, ``seed``, ``use_class_weights``.

    Returns:
        The fitted :class:`~sklearn.linear_model.LogisticRegression`.
    """
    run_config = dict(hyperparameters)
    run_config.setdefault("seed",              42)
    run_config.setdefault("use_class_weights", True)
    run_config.setdefault("solver",            "saga")
    run_config.setdefault("max_iter",          1000)

    logger.info("Play called with hyperparameters: %s", run_config)

    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
    dir_name  = str(RESULTS_DIR / f"results_lr_{timestamp}")
    os.makedirs(dir_name, exist_ok=True)

    with open(os.path.join(dir_name, "hyperparameters.txt"), "wt") as f:
        f.write("Hyperparameters:\n")
        for key, value in run_config.items():
            f.write(f"\t{key}: {value}\n")

    window_size: int = int(run_config["window_size"])

    train_ds = TractorActivityDataset(
        DATA_DIR / "train_day_tractor.xlsx", window_size=window_size
    )
    val_ds = TractorActivityDataset(
        DATA_DIR / "val_day_tractor.xlsx",
        window_size=window_size,
        label_encoder=train_ds.label_encoder,
    )
    test_ds = TractorActivityDataset(
        DATA_DIR / "test_day_tractor.xlsx",
        window_size=window_size,
        label_encoder=train_ds.label_encoder,
    )

    if verbose:
        logger.info("Classes: %s", train_ds.classes)
        logger.info(
            "Windows — train: %d  val: %d  test: %d",
            len(train_ds), len(val_ds), len(test_ds),
        )

    X_train, y_train = _extract_arrays(train_ds)
    X_val,   y_val   = _extract_arrays(val_ds)
    X_test,  y_test  = _extract_arrays(test_ds)

    class_weight = "balanced" if run_config["use_class_weights"] else None

    model = LogisticRegression(
        penalty      = run_config["penalty"],
        C            = float(run_config["C"]),
        l1_ratio     = float(run_config["l1_ratio"]),
        solver       = run_config["solver"],
        max_iter     = int(run_config["max_iter"]),
        class_weight = class_weight,
        random_state = int(run_config["seed"]),
        n_jobs       = -1,
    )

    fit_start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time  = time.perf_counter() - fit_start
    n_iter    = int(model.n_iter_.max())
    converged = n_iter < int(run_config["max_iter"])
    logger.info(
        "Model fitted in %.3f seconds  (iterations=%d, converged=%s)",
        fit_time, n_iter, converged,
    )

    original_labels = train_ds.classes
    n_classes       = train_ds.n_classes
    logger.info("Class labels: %s", original_labels)

    y_pred_train  = model.predict(X_train)
    y_pred_val    = model.predict(X_val)
    y_pred_test   = model.predict(X_test)

    y_probs_train = _align_proba(model.predict_proba(X_train), model.classes_, n_classes)
    y_probs_val   = _align_proba(model.predict_proba(X_val),   model.classes_, n_classes)
    y_probs_test  = _align_proba(model.predict_proba(X_test),  model.classes_, n_classes)

    train_metrics = _compute_metrics(y_train, y_pred_train, y_probs_train)
    val_metrics   = _compute_metrics(y_val,   y_pred_val,   y_probs_val)
    test_metrics  = _compute_metrics(y_test,  y_pred_test,  y_probs_test)

    for split, metrics in [("Train", train_metrics), ("Val", val_metrics), ("Test", test_metrics)]:
        logger.info(
            "%s — acc: %.4f  prec: %.4f  rec: %.4f  f1: %.4f  auc: %.4f",
            split,
            metrics["accuracy"], metrics["precision"],
            metrics["recall"],   metrics["f1"], metrics["auc"],
        )

    summary_metrics = {
        "fit_time_seconds": fit_time,
        "n_iter":           n_iter,
        "converged":        converged,
        **{f"train_{k}": v for k, v in train_metrics.items()},
        **{f"val_{k}":   v for k, v in val_metrics.items()},
        **{f"test_{k}":  v for k, v in test_metrics.items()},
    }

    with open(os.path.join(dir_name, "hyperparameters.txt"), "at") as f:
        f.write("\nRun summary:\n")
        for key, value in summary_metrics.items():
            f.write(f"\t{key}: {value}\n")

    pd.DataFrame([summary_metrics]).to_csv(
        os.path.join(dir_name, "run_summary.csv"), index=False
    )

    with open(os.path.join(dir_name, "classification_report.txt"), "wt") as f:
        for split, metrics, y_true, y_pred in [
            ("Train", train_metrics, y_train, y_pred_train),
            ("Val",   val_metrics,   y_val,   y_pred_val),
            ("Test",  test_metrics,  y_test,  y_pred_test),
        ]:
            f.write(f"[{split}]\n")
            f.write(f"accuracy:  {metrics['accuracy']:.6f}\n")
            f.write(f"precision: {metrics['precision']:.6f}\n")
            f.write(f"recall:    {metrics['recall']:.6f}\n")
            f.write(f"f1:        {metrics['f1']:.6f}\n")
            f.write(f"auc:       {metrics['auc']:.6f}\n\n")
            report = classification_report(
                y_true, y_pred,
                labels=np.arange(n_classes),
                target_names=original_labels,
                output_dict=False,
                zero_division=0,
            )
            logger.info("%s Classification Report:\n%s", split, report)
            f.write("Classification Report:\n")
            f.write(report)
            f.write("\n")

    cm = confusion_matrix(y_test, y_pred_test)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, cmap="Reds", fmt="d",
        xticklabels=original_labels.tolist(),
        yticklabels=original_labels.tolist(),
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix — Test set")
    plt.savefig(os.path.join(dir_name, "confusion_matrix.png"), bbox_inches="tight")
    if save_eps:
        plt.savefig(os.path.join(dir_name, "confusion_matrix.eps"), bbox_inches="tight")
    plt.close()

    all_results_path = str(RESULTS_DIR / "all_results_lr.xlsx")
    summary_keys     = list(summary_metrics.keys())
    all_results_cols = ["timestamp"] + list(run_config.keys()) + summary_keys

    if os.path.exists(all_results_path):
        df_results = pd.read_excel(all_results_path)
        for col in all_results_cols:
            if col not in df_results.columns:
                df_results[col] = pd.NA
    else:
        df_results = pd.DataFrame(columns=all_results_cols)

    row: dict[str, Any] = {"timestamp": timestamp}
    row.update(run_config)
    row.update(summary_metrics)
    df_results.loc[len(df_results)] = row
    df_results = df_results[all_results_cols]
    df_results.to_excel(all_results_path, index=False)

    logger.info("Writing predictions...")
    write_predictions(X_train, y_train, y_pred_train, dir_name, "train")
    write_predictions(X_val,   y_val,   y_pred_val,   dir_name, "val")
    write_predictions(X_test,  y_test,  y_pred_test,  dir_name, "test")

    return model


if __name__ == "__main__":
    setup_logging()

    sampler = ParameterSampler(HP_DISTRIBUTIONS, n_iter=N_TRIALS, random_state=SEARCH_SEED)
    for trial, params in enumerate(sampler, start=1):
        full_params = {**HP_FIXED, **params}
        logger.info("Trial %d/%d — hyperparameters: %s", trial, N_TRIALS, full_params)
        play(verbose=(trial == 1), **full_params)
