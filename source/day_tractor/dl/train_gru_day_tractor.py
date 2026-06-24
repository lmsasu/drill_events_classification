"""
GRU activity classifier for the day-tractor dataset.

Training data is loaded from the three pre-split Excel files produced by
``split_dataset.py``.  A random hyper-parameter search is performed over
``N_TRIALS`` configurations; results are appended to a shared summary
spreadsheet for cross-trial comparison.

Key differences from the continuous-dataset GRU trainer
---------------------------------------------------------
* Splits are pre-computed (three separate Excel files).
* Windows are extracted per-segment by :class:`TractorActivityDataset`;
  ``window_size`` is therefore a hyper-parameter that triggers re-creation
  of the datasets on every trial.
* ``CrossEntropyLoss`` is optionally weighted by inverse class frequency to
  compensate for the imbalance in the ``Task`` column.
"""

import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from scipy.stats import loguniform, randint, uniform
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterSampler
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from day_tractor.data.dataset import TractorActivityDataset
from shared_modules.logging_setup import setup_logging
from shared_modules.config import OPTIMIZATION_TRIALS
from shared_modules.timing import timeit

logger = logging.getLogger(__name__)

# ── paths ─────────────────────────────────────────────────────────────────────
DATA_DIR    = Path(__file__).resolve().parents[3] / "data" / "day_tractor"
RESULTS_DIR = Path(__file__).resolve().parents[3] / "results" / "day_tractor" / "dl" / "GRU"

# ── random search configuration ───────────────────────────────────────────────
N_TRIALS: int = OPTIMIZATION_TRIALS
SEARCH_SEED: int | None = 0

# Sampled hyperparameters - use scipy.stats distributions or lists:
#     randint(a, b)        uniform integer  in [a, b-1]
#     uniform(loc, scale)  uniform float    in [loc, loc+scale]
#     loguniform(a, b)     log-uniform float in [a, b]
#     [v1, v2, ...]        uniform draw from list
HP_DISTRIBUTIONS: dict[str, Any] = {
    "window_size":   randint(10, 51),
    "batch_size":    [16, 32, 64],
    "hidden_size":   [64, 128, 256],
    "num_layers":    randint(1, 5),
    "dropout":       uniform(0.1, 0.4),
    "learning_rate": loguniform(1e-4, 1e-2),
}

# Fixed hyperparameters - passed unchanged to every trial.
HP_FIXED: dict[str, Any] = {
    "epochs":                   100,
    "seed":                     42,
    "deterministic":            True,
    "benchmark":                False,
    "early_stopping_patience":  8,
    "early_stopping_min_delta": 0.0,
    "grad_clip_norm":           1.0,
    "use_class_weights":        True,   # weight CrossEntropyLoss by inverse class freq
}
# ─────────────────────────────────────────────────────────────────────────────


def set_reproducibility(seed: int, deterministic: bool, benchmark: bool) -> None:
    """Set RNG seeds and cuDNN flags for reproducible experiments.

    Args:
        seed: Integer seed applied to Python, NumPy, and PyTorch RNGs.
        deterministic: If ``True``, forces cuDNN to use deterministic
            algorithms (may reduce throughput).
        benchmark: If ``True``, enables cuDNN auto-tuning (incompatible with
            deterministic mode; overridden to ``False`` when both are set).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic and benchmark:
        logger.warning(
            "deterministic=True is incompatible with benchmark=True; forcing benchmark=False"
        )
        benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark


class GRUClassifier(nn.Module):
    """Stacked GRU followed by a linear classification head.

    The last hidden state of the top GRU layer is fed into a fully-connected
    layer that produces one logit per class.  Unlike LSTM, GRU has no separate
    cell state, so the forward pass unpacks only ``h_n``.

    Args:
        input_size: Number of input features at each time step.
        hidden_size: Number of hidden units in each GRU layer.
        num_layers: Number of stacked GRU layers.
        num_classes: Number of output classes.
        dropout: Dropout probability applied between GRU layers (ignored
            when ``num_layers == 1``).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            x: Input tensor of shape ``(batch, window_size, input_size)``.

        Returns:
            Logits tensor of shape ``(batch, num_classes)``.
        """
        _, h_n = self.gru(x)
        return self.fc(h_n[-1])


def make_loaders(
    train_ds: TractorActivityDataset,
    val_ds: TractorActivityDataset,
    test_ds: TractorActivityDataset,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Wrap dataset instances in reproducible DataLoaders.

    Args:
        train_ds: Training split dataset.
        val_ds: Validation split dataset.
        test_ds: Test split dataset.
        batch_size: Mini-batch size for all loaders.
        seed: Seed for the DataLoader generator (worker RNG initialisation).

    Returns:
        A tuple ``(train_loader, val_loader, test_loader)``.
    """

    def seed_worker(_: int) -> None:
        worker_seed = torch.initial_seed() % (2**32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    use_cuda = torch.cuda.is_available()
    num_workers = max(1, min(4, (os.cpu_count() or 1))) if use_cuda else 0
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader_kwargs = {
        "batch_size":         batch_size,
        "pin_memory":         use_cuda,
        "num_workers":        num_workers,
        "persistent_workers": num_workers > 0,
        "worker_init_fn":     seed_worker,
        "generator":          generator,
    }
    return (
        DataLoader(train_ds, shuffle=True,  **loader_kwargs),
        DataLoader(val_ds,   shuffle=False, **loader_kwargs),
        DataLoader(test_ds,  shuffle=False, **loader_kwargs),
    )


def run_epoch(
    model: GRUClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | str = "cpu",
    grad_clip_norm: float | None = None,
) -> tuple[float, float]:
    """Run one full pass over a data loader.

    When *optimizer* is provided the model is set to training mode and
    gradients are updated; otherwise the model runs in evaluation mode with
    gradient computation disabled.

    Args:
        model: The GRU classifier.
        loader: DataLoader yielding ``(x, y)`` batches where ``y`` is a
            1-D long tensor of class indices.
        criterion: Loss function (e.g. ``CrossEntropyLoss``).
        optimizer: Optimizer for the training pass; ``None`` for evaluation.
        device: Device to move batches to before the forward pass.
        grad_clip_norm: If positive, clip gradient norms to this value.

    Returns:
        A tuple ``(mean_loss, accuracy)`` over the full loader.
    """
    training = optimizer is not None
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            if training:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=grad_clip_norm
                    )
                optimizer.step()
            total_loss += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(yb)
    return total_loss / total, correct / total


def _collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device | str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference over a loader and return true labels, predicted classes, and softmax probs.

    Args:
        model: Trained classifier.
        loader: DataLoader yielding ``(x, y)`` pairs.
        device: Inference device.

    Returns:
        Tuple ``(y_true, y_pred, y_probs)`` as numpy arrays.
    """
    model.eval()
    all_true, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(device, non_blocking=True))
            all_true.append(yb.numpy())
            all_preds.append(logits.argmax(1).cpu().numpy())
            all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return (
        np.concatenate(all_true),
        np.concatenate(all_preds),
        np.concatenate(all_probs),
    )


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
) -> dict[str, float]:
    """Return accuracy, weighted and macro precision / recall / F1, and weighted AUC.

    AUC is set to ``nan`` when not all classes are present in ``y_true``
    (e.g. a split whose segments are all shorter than the current
    ``window_size``, so an entire class has zero windows).

    Args:
        y_true: Ground-truth class indices.
        y_pred: Predicted class indices.
        y_probs: Softmax probabilities, shape ``(n, n_classes)``.

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


def plot_history(
    history: dict[str, list[float]],
    dir_name: str,
    save_eps: bool = True,
) -> None:
    """Save loss and accuracy curves as PNG (and optionally EPS) files.

    Args:
        history: Dict with keys ``train_loss``, ``val_loss``,
            ``train_accuracy``, ``val_accuracy``; each value is a list of
            per-epoch scalars.
        dir_name: Directory where plot files are written.
        save_eps: When ``True``, also save an EPS copy of each figure.
    """
    os.makedirs(dir_name, exist_ok=True)
    epoch_range = range(1, len(history["train_loss"]) + 1)
    for metric in ("loss", "accuracy"):
        plt.figure(figsize=(12, 6))
        plt.xticks(epoch_range)
        plt.plot(epoch_range, history[f"train_{metric}"], label=f"train_{metric}")
        plt.plot(epoch_range, history[f"val_{metric}"],   label=f"val_{metric}")
        plt.title(f"Training vs Validation {metric}")
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.legend()
        plt.savefig(os.path.join(dir_name, f"{metric}.png"), bbox_inches="tight")
        if save_eps:
            plt.savefig(os.path.join(dir_name, f"{metric}.eps"), bbox_inches="tight")
        plt.close()


def write_predictions(
    model: GRUClassifier,
    loader: DataLoader,
    dir_name: str,
    subset_type: str,
    device: torch.device | str,
) -> None:
    """Write an Excel file with model predictions for a data subset.

    Each row contains the flattened input window, the predicted class index,
    and the ground-truth class index.

    Args:
        model: Trained GRU classifier.
        loader: DataLoader for the subset.
        dir_name: Directory where the Excel file is written.
        subset_type: Label used in the output filename (e.g. ``"train"``).
        device: Device to run inference on.
    """
    model.eval()
    all_x, all_preds, all_true = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            all_preds.append(model(xb.to(device, non_blocking=True)).argmax(1).cpu().numpy())
            all_x.append(xb.numpy())
            all_true.append(yb.numpy())
    x_arr = np.concatenate(all_x)
    df_out = pd.DataFrame(x_arr.reshape(x_arr.shape[0], -1))
    df_out["predicted"]     = np.concatenate(all_preds)
    df_out["y_groundtruth"] = np.concatenate(all_true)
    df_out.to_excel(os.path.join(dir_name, f"{subset_type}_predictions.xlsx"), index=False)


@timeit
def play(
    verbose: bool = True,
    save_eps: bool = True,
    **hyperparameters: Any,
) -> tuple[dict[str, list[float]], GRUClassifier]:
    """Train and evaluate the GRU classifier on the day-tractor splits.

    Loads the three pre-split Excel files, builds datasets with the trial's
    ``window_size``, trains the model, and writes metrics, plots, and
    predictions to a timestamped results sub-directory.

    Args:
        verbose: When ``True``, log dataset statistics.
        save_eps: When ``True``, also save EPS versions of all figures.
        **hyperparameters: Training configuration.  Expected keys:
            ``window_size``, ``batch_size``, ``epochs``, ``hidden_size``,
            ``num_layers``, ``dropout``, ``learning_rate``,
            ``use_class_weights``.

    Returns:
        A tuple ``(history, model)``.
    """
    run_config = dict(hyperparameters)
    run_config.setdefault("seed",                     42)
    run_config.setdefault("deterministic",            True)
    run_config.setdefault("benchmark",                False)
    run_config.setdefault("early_stopping_patience",  8)
    run_config.setdefault("early_stopping_min_delta", 0.0)
    run_config.setdefault("grad_clip_norm",           1.0)
    run_config.setdefault("use_class_weights",        True)

    set_reproducibility(
        seed=int(run_config["seed"]),
        deterministic=bool(run_config["deterministic"]),
        benchmark=bool(run_config["benchmark"]),
    )

    logger.info("Play called with hyperparameters: %s", run_config)

    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
    dir_name  = str(RESULTS_DIR / f"results_gru_{timestamp}")
    os.makedirs(dir_name, exist_ok=True)

    with open(os.path.join(dir_name, "hyperparameters.txt"), "wt") as f:
        f.write("Hyperparameters:\n")
        for key, value in run_config.items():
            f.write(f"\t{key}: {value}\n")
        f.write("\nRuntime flags:\n")
        f.write(f"\tcudnn_deterministic: {torch.backends.cudnn.deterministic}\n")
        f.write(f"\tcudnn_benchmark: {torch.backends.cudnn.benchmark}\n")

    window_size: int = int(run_config["window_size"])

    # Build datasets; val and test reuse the train label encoder so that
    # class indices are identical across all three splits.
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    cuda_available  = torch.cuda.is_available()
    gpu_name        = torch.cuda.get_device_name(0) if cuda_available else "cpu"
    cudnn_enabled   = torch.backends.cudnn.enabled
    cudnn_version   = torch.backends.cudnn.version()

    logger.info("CUDA available: %s", cuda_available)
    logger.info("GPU: %s", gpu_name)
    logger.info("cuDNN enabled: %s  version: %s", cudnn_enabled, cudnn_version)

    train_loader, val_loader, test_loader = make_loaders(
        train_ds, val_ds, test_ds,
        batch_size=int(run_config["batch_size"]),
        seed=int(run_config["seed"]),
    )

    model = GRUClassifier(
        input_size  = train_ds.n_features,
        hidden_size = int(run_config["hidden_size"]),
        num_layers  = int(run_config["num_layers"]),
        num_classes = train_ds.n_classes,
        dropout     = float(run_config["dropout"]),
    ).to(device)

    class_weights = (
        train_ds.class_weights().to(device)
        if run_config["use_class_weights"]
        else None
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(run_config["learning_rate"]))

    history: dict[str, list[float]] = {
        "train_loss": [], "val_loss": [],
        "train_accuracy": [], "val_accuracy": [],
    }
    epochs: int                = int(run_config["epochs"])
    patience: int              = int(run_config["early_stopping_patience"])
    min_delta: float           = float(run_config["early_stopping_min_delta"])
    grad_clip_norm: float      = float(run_config["grad_clip_norm"])
    best_val_loss              = float("inf")
    epochs_without_improvement = 0
    best_model_state: dict[str, torch.Tensor] | None = None
    checkpoint_path            = os.path.join(dir_name, "best_model.pt")
    stopped_early              = False
    train_start_time           = time.perf_counter()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device,
            grad_clip_norm=grad_clip_norm,
        )
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device=device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_acc)
        history["val_accuracy"].append(val_acc)
        logger.info(
            "Epoch %d/%d — loss: %.4f  acc: %.4f  val_loss: %.4f  val_acc: %.4f",
            epoch, epochs, train_loss, train_acc, val_loss, val_acc,
        )

        if val_loss < (best_val_loss - min_delta):
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_model_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            torch.save(best_model_state, checkpoint_path)
        else:
            epochs_without_improvement += 1

        if patience > 0 and epochs_without_improvement >= patience:
            stopped_early = True
            logger.info(
                "Early stopping at epoch %d (patience=%d, min_delta=%.6f)",
                epoch, patience, min_delta,
            )
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info("Restored best checkpoint from %s", checkpoint_path)

    total_train_time = time.perf_counter() - train_start_time
    best_epoch_idx   = int(np.argmin(history["val_loss"]))
    best_epoch       = best_epoch_idx + 1

    pd.DataFrame(history).to_csv(os.path.join(dir_name, "training_log.csv"), index=False)

    original_labels = train_ds.classes
    logger.info("Class labels: %s", original_labels)

    train_loss, _ = run_epoch(model, train_loader, criterion, device=device)
    val_loss,   _ = run_epoch(model, val_loader,   criterion, device=device)
    test_loss,  _ = run_epoch(model, test_loader,  criterion, device=device)

    y_true_train, y_pred_train, y_probs_train = _collect_predictions(model, train_loader, device)
    y_true_val,   y_pred_val,   y_probs_val   = _collect_predictions(model, val_loader,   device)
    y_true_test,  y_pred_test,  y_probs_test  = _collect_predictions(model, test_loader,  device)

    train_metrics = _compute_metrics(y_true_train, y_pred_train, y_probs_train)
    val_metrics   = _compute_metrics(y_true_val,   y_pred_val,   y_probs_val)
    test_metrics  = _compute_metrics(y_true_test,  y_pred_test,  y_probs_test)

    for split, metrics in [("Train", train_metrics), ("Val", val_metrics), ("Test", test_metrics)]:
        logger.info(
            "%s — acc: %.4f  prec: %.4f  rec: %.4f  f1: %.4f  auc: %.4f",
            split,
            metrics["accuracy"], metrics["precision"],
            metrics["recall"],   metrics["f1"], metrics["auc"],
        )

    summary_metrics = {
        "best_epoch":               best_epoch,
        "best_val_loss":            float(history["val_loss"][best_epoch_idx]),
        "train_loss":               train_loss,
        **{f"train_{k}": v for k, v in train_metrics.items()},
        "val_loss":                 val_loss,
        **{f"val_{k}":   v for k, v in val_metrics.items()},
        "test_loss":                test_loss,
        **{f"test_{k}":  v for k, v in test_metrics.items()},
        "stopped_early":            stopped_early,
        "epochs_ran":               len(history["train_loss"]),
        "total_train_time_seconds": total_train_time,
        "cuda_available":           cuda_available,
        "gpu_name":                 gpu_name,
        "cudnn_enabled":            cudnn_enabled,
        "cudnn_version":            cudnn_version,
        "cudnn_deterministic":      torch.backends.cudnn.deterministic,
        "cudnn_benchmark":          torch.backends.cudnn.benchmark,
    }

    with open(os.path.join(dir_name, "hyperparameters.txt"), "at") as f:
        f.write("\nRun summary:\n")
        for key, value in summary_metrics.items():
            f.write(f"\t{key}: {value}\n")

    pd.DataFrame([summary_metrics]).to_csv(
        os.path.join(dir_name, "run_summary.csv"), index=False
    )

    plot_history(history, dir_name, save_eps=save_eps)

    with open(os.path.join(dir_name, "classification_report.txt"), "wt") as f:
        for split, loss, metrics, y_true, y_pred in [
            ("Train", train_loss, train_metrics, y_true_train, y_pred_train),
            ("Val",   val_loss,   val_metrics,   y_true_val,   y_pred_val),
            ("Test",  test_loss,  test_metrics,  y_true_test,  y_pred_test),
        ]:
            f.write(f"[{split}]\n")
            f.write(f"loss:      {loss:.6f}\n")
            f.write(f"accuracy:  {metrics['accuracy']:.6f}\n")
            f.write(f"precision: {metrics['precision']:.6f}\n")
            f.write(f"recall:    {metrics['recall']:.6f}\n")
            f.write(f"f1:        {metrics['f1']:.6f}\n")
            f.write(f"auc:       {metrics['auc']:.6f}\n\n")
            report = classification_report(
                y_true, y_pred,
                labels=np.arange(len(original_labels)),
                target_names=original_labels,
                output_dict=False,
                zero_division=0,
            )
            logger.info("%s Classification Report:\n%s", split, report)
            f.write("Classification Report:\n")
            f.write(report)
            f.write("\n")

    cm = confusion_matrix(y_true_test, y_pred_test)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, cmap="Reds", fmt="d",
        xticklabels=original_labels.tolist(),
        yticklabels=original_labels.tolist(),
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix — Test set")
    _cm_filename = "confusion_matrix"
    plt.savefig(os.path.join(dir_name, f"{_cm_filename}.png"), bbox_inches="tight")
    if save_eps:
        plt.savefig(os.path.join(dir_name, f"{_cm_filename}.eps"), bbox_inches="tight")
    plt.close()

    all_results_path = str(RESULTS_DIR / "all_results_gru.xlsx")
    summary_keys     = list(summary_metrics.keys())
    all_results_cols = ["model", "timestamp"] + list(run_config.keys()) + summary_keys

    if os.path.exists(all_results_path):
        df_results = pd.read_excel(all_results_path)
        for col in all_results_cols:
            if col not in df_results.columns:
                df_results[col] = pd.NA
    else:
        df_results = pd.DataFrame(columns=all_results_cols)

    row: dict[str, Any] = {"model": "GRU", "timestamp": timestamp}
    row.update(run_config)
    row.update({m: history[m][-1] for m in history})
    row.update(summary_metrics)
    df_results.loc[len(df_results)] = row
    df_results = df_results[all_results_cols]
    df_results.to_excel(all_results_path, index=False)

    logger.info("Writing predictions...")
    write_predictions(model, train_loader, dir_name, "train", device)
    write_predictions(model, val_loader,   dir_name, "val",   device)
    write_predictions(model, test_loader,  dir_name, "test",  device)

    return history, model


if __name__ == "__main__":
    setup_logging()
    logger.info("Torch version: %s", torch.__version__)
    logger.info("CUDA available: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    sampler = ParameterSampler(HP_DISTRIBUTIONS, n_iter=N_TRIALS, random_state=SEARCH_SEED)
    for trial, params in enumerate(sampler, start=1):
        full_params = {**HP_FIXED, **params}
        logger.info("Trial %d/%d — hyperparameters: %s", trial, N_TRIALS, full_params)
        play(verbose=(trial == 1), **full_params)
