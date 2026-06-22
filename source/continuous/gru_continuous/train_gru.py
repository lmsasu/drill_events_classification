import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterSampler
from sklearn.preprocessing import LabelEncoder
from scipy.stats import loguniform, randint, uniform

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from continuous.data.data_preparation import load_dataset, prepare_data
from shared_modules.logging_setup import setup_logging
from shared_modules.timing import timeit

logger = logging.getLogger(__name__)


# -- Random search configuration -----------------------------------------------
# Set the number of trials and the search RNG seed here.
N_TRIALS: int = 30
SEARCH_SEED: int | None = 0  # set to None for a non-reproducible search

# Sampled hyperparameters - use scipy.stats distributions or lists:
#     randint(a, b)        uniform integer  in [a, b-1]
#     uniform(loc, scale)  uniform float    in [loc, loc+scale]
#     loguniform(a, b)     log-uniform float in [a, b]
#     [v1, v2, ...]        uniform draw from list
HP_DISTRIBUTIONS: dict[str, Any] = {
    "lookback_window": randint(10, 51),
    "batch_size": [16, 32, 64],
    "hidden_size": [64, 128, 256],
    "num_layers": randint(1, 5),
    "dropout": uniform(0.1, 0.4),
    "learning_rate": loguniform(1e-4, 1e-2),
}

# Fixed hyperparameters - passed unchanged to every trial.
HP_FIXED: dict[str, Any] = {
    "epochs": 100,
    "seed": 42,
    "deterministic": True,
    "benchmark": False,
    "early_stopping_patience": 8,
    "early_stopping_min_delta": 0.0,
    "grad_clip_norm": 1.0,
}
# -- End of random search configuration ----------------------------------------


def set_reproducibility(seed: int, deterministic: bool, benchmark: bool) -> None:
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
        _, h_n = self.gru(x)
        return self.fc(h_n[-1])


def make_loaders(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    def to_dataset(x: np.ndarray, y: np.ndarray) -> TensorDataset:
        return TensorDataset(
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(np.argmax(y, axis=1), dtype=torch.long),
        )

    def seed_worker(_: int) -> None:
        worker_seed = torch.initial_seed() % (2**32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    use_cuda = torch.cuda.is_available()
    num_workers = max(1, min(4, (os.cpu_count() or 1))) if use_cuda else 0
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader_kwargs = {
        "batch_size": batch_size,
        "pin_memory": use_cuda,
        "num_workers": num_workers,
        "persistent_workers": num_workers > 0,
        "worker_init_fn": seed_worker,
        "generator": generator,
    }

    return (
        DataLoader(to_dataset(x_train, y_train), shuffle=True, **loader_kwargs),
        DataLoader(to_dataset(x_val, y_val), shuffle=False, **loader_kwargs),
        DataLoader(to_dataset(x_test, y_test), shuffle=False, **loader_kwargs),
    )


def run_epoch(
    model: GRUClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | str = "cpu",
    grad_clip_norm: float | None = None,
) -> tuple[float, float]:
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


def plot_history(
    history: dict[str, list[float]],
    dir_name: str,
    save_eps: bool = True,
) -> None:
    os.makedirs(dir_name, exist_ok=True)
    epoch_range = range(1, len(history["train_loss"]) + 1)
    for metric in ("loss", "accuracy"):
        plt.figure(figsize=(12, 6))
        plt.xticks(epoch_range)
        plt.plot(epoch_range, history[f"train_{metric}"], label=f"train_{metric}")
        plt.plot(epoch_range, history[f"val_{metric}"], label=f"val_{metric}")
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
    y_onehot: np.ndarray,
    dir_name: str,
    subset_type: str,
    device: torch.device | str,
) -> None:
    model.eval()
    all_x, all_preds = [], []
    with torch.no_grad():
        for xb, _ in loader:
            all_preds.append(
                model(xb.to(device, non_blocking=True)).argmax(1).cpu().numpy()
            )
            all_x.append(xb.numpy())
    x_arr = np.concatenate(all_x)
    df_out = pd.DataFrame(x_arr.reshape(x_arr.shape[0], -1))
    df_out["predicted"] = np.concatenate(all_preds)
    df_out["y_groundtruth"] = np.argmax(y_onehot, axis=1)
    df_out.to_excel(
        os.path.join(dir_name, f"{subset_type}_predictions.xlsx"), index=False
    )


def _collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device | str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference over a loader and return true labels, predicted classes, and softmax probs."""
    model.eval()
    all_true, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(device, non_blocking=True))
            all_true.append(yb.numpy())
            all_preds.append(logits.argmax(1).cpu().numpy())
            all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(all_true), np.concatenate(all_preds), np.concatenate(all_probs)


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
) -> dict[str, float]:
    """Return accuracy, weighted and macro precision/recall/F1, and weighted AUC."""
    return {
        "accuracy":        float((y_pred == y_true).mean()),
        "precision":       float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall":          float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1":              float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "auc":             float(roc_auc_score(y_true, y_probs, multi_class="ovr", average="weighted")),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall":    float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1":        float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


@timeit
def play(
    df: pd.DataFrame,
    verbose: bool = True,
    save_eps: bool = True,
    **hyperparameters: Any,
) -> tuple[dict[str, list[float]], np.ndarray, np.ndarray, np.ndarray, GRUClassifier]:
    run_config = dict(hyperparameters)
    run_config.setdefault("seed", 42)
    run_config.setdefault("deterministic", True)
    run_config.setdefault("benchmark", False)
    run_config.setdefault("early_stopping_patience", 8)
    run_config.setdefault("early_stopping_min_delta", 0.0)
    run_config.setdefault("grad_clip_norm", 1.0)

    set_reproducibility(
        seed=int(run_config["seed"]),
        deterministic=bool(run_config["deterministic"]),
        benchmark=bool(run_config["benchmark"]),
    )

    logger.info("Play function called with hyperparameters: %s", run_config)

    dir_root_results = "results"
    timestamp = str(pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S"))
    dir_name = os.path.join(dir_root_results, f"results_gru_{timestamp}")
    os.makedirs(dir_name, exist_ok=True)

    with open(os.path.join(dir_name, "hyperparameters.txt"), "wt") as f:
        f.write("Hyperparameters:\n")
        for key, value in run_config.items():
            f.write(f"\t{key}: {value}\n")
        f.write("\nRuntime flags:\n")
        f.write(f"\tcudnn_deterministic: {torch.backends.cudnn.deterministic}\n")
        f.write(f"\tcudnn_benchmark: {torch.backends.cudnn.benchmark}\n")

    lookback_window: int = run_config["lookback_window"]
    x_train, x_val, x_test, y_train, y_val, y_test, label_encoder = prepare_data(
        df, lookback_window, verbose=verbose
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "cpu"
    cudnn_enabled = torch.backends.cudnn.enabled
    cudnn_version = torch.backends.cudnn.version()

    logger.info("CUDA available: %s", cuda_available)
    logger.info("GPU name: %s", gpu_name)
    logger.info("cuDNN enabled: %s", cudnn_enabled)
    logger.info("cuDNN version: %s", cudnn_version)

    train_loader, val_loader, test_loader = make_loaders(
        x_train,
        x_val,
        x_test,
        y_train,
        y_val,
        y_test,
        run_config["batch_size"],
        seed=int(run_config["seed"]),
    )

    model = GRUClassifier(
        input_size=x_train.shape[2],
        hidden_size=run_config["hidden_size"],
        num_layers=run_config["num_layers"],
        num_classes=y_train.shape[1],
        dropout=run_config["dropout"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=run_config["learning_rate"])

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }
    epochs: int = run_config["epochs"]
    early_stopping_patience = int(run_config["early_stopping_patience"])
    early_stopping_min_delta = float(run_config["early_stopping_min_delta"])
    grad_clip_norm = float(run_config["grad_clip_norm"])
    best_val_loss_for_early_stop = float("inf")
    epochs_without_improvement = 0
    best_model_state: dict[str, torch.Tensor] | None = None
    checkpoint_path = os.path.join(dir_name, "best_model.pt")
    stopped_early = False
    train_start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            grad_clip_norm=grad_clip_norm,
        )
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device=device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_acc)
        history["val_accuracy"].append(val_acc)
        logger.info(
            "Epoch %d/%d - loss: %.4f, acc: %.4f, val_loss: %.4f, val_acc: %.4f",
            epoch,
            epochs,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
        )

        if val_loss < (best_val_loss_for_early_stop - early_stopping_min_delta):
            best_val_loss_for_early_stop = val_loss
            epochs_without_improvement = 0
            best_model_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            torch.save(best_model_state, checkpoint_path)
        else:
            epochs_without_improvement += 1

        if (
            early_stopping_patience > 0
            and epochs_without_improvement >= early_stopping_patience
        ):
            stopped_early = True
            logger.info(
                "Early stopping at epoch %d (patience=%d, min_delta=%.6f)",
                epoch,
                early_stopping_patience,
                early_stopping_min_delta,
            )
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info("Restored best checkpoint from %s", checkpoint_path)

    total_train_time_seconds = time.perf_counter() - train_start_time
    best_epoch_idx = int(np.argmin(history["val_loss"]))
    best_epoch = best_epoch_idx + 1
    best_val_loss = float(history["val_loss"][best_epoch_idx])
    best_val_accuracy = float(history["val_accuracy"][best_epoch_idx])

    logger.info("Best epoch (by val_loss): %d", best_epoch)
    logger.info("Best val_loss: %.6f", best_val_loss)
    logger.info("Best val_accuracy: %.6f", best_val_accuracy)
    logger.info("Total train time (s): %.3f", total_train_time_seconds)

    pd.DataFrame(history).to_csv(
        os.path.join(dir_name, "training_log.csv"), index=False
    )

    original_labels: np.ndarray = label_encoder.inverse_transform(
        np.arange(len(label_encoder.classes_))
    )
    logger.info("Original Labels: %s", original_labels)

    train_loss, _ = run_epoch(model, train_loader, criterion, device=device)
    val_loss, _ = run_epoch(model, val_loader, criterion, device=device)
    test_loss, _ = run_epoch(model, test_loader, criterion, device=device)

    y_true_train, y_pred_train, y_probs_train = _collect_predictions(model, train_loader, device)
    y_true_val, y_pred_val, y_probs_val = _collect_predictions(model, val_loader, device)
    y_true_test, y_pred_test, y_probs_test = _collect_predictions(model, test_loader, device)

    train_metrics = _compute_metrics(y_true_train, y_pred_train, y_probs_train)
    val_metrics = _compute_metrics(y_true_val, y_pred_val, y_probs_val)
    test_metrics = _compute_metrics(y_true_test, y_pred_test, y_probs_test)

    for split, metrics in [("Train", train_metrics), ("Val", val_metrics), ("Test", test_metrics)]:
        logger.info(
            "%s - acc: %.4f  prec: %.4f  rec: %.4f  f1: %.4f  auc: %.4f",
            split,
            metrics["accuracy"], metrics["precision"], metrics["recall"],
            metrics["f1"], metrics["auc"],
        )

    summary_metrics = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "train_loss": train_loss,
        **{f"train_{k}": v for k, v in train_metrics.items()},
        "val_loss": val_loss,
        **{f"val_{k}": v for k, v in val_metrics.items()},
        "test_loss": test_loss,
        **{f"test_{k}": v for k, v in test_metrics.items()},
        "stopped_early": stopped_early,
        "epochs_ran": len(history["train_loss"]),
        "total_train_time_seconds": total_train_time_seconds,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "cudnn_enabled": cudnn_enabled,
        "cudnn_version": cudnn_version,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
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
            report = classification_report(y_true, y_pred, target_names=original_labels, output_dict=False)
            logger.info("%s Classification Report:\n%s", split, report)
            f.write("Classification Report:\n")
            f.write(report)
            f.write("\n")

    cm = confusion_matrix(y_true_test, y_pred_test)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        cmap="Reds",
        fmt="d",
        xticklabels=original_labels.tolist(),
        yticklabels=original_labels.tolist(),
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix on Test Data")
    _cm_filename = "confusion_matrix"
    plt.savefig(os.path.join(dir_name, f"{_cm_filename}.png"), bbox_inches="tight")
    if save_eps:
        plt.savefig(os.path.join(dir_name, f"{_cm_filename}.eps"), bbox_inches="tight")
    plt.close()

    all_results_path = os.path.join(dir_root_results, "all_results_gru.xlsx")
    summary_keys = list(summary_metrics.keys())
    all_results_columns = ["timestamp"] + list(run_config.keys()) + summary_keys

    if os.path.exists(all_results_path):
        df_results = pd.read_excel(all_results_path)
        for col in all_results_columns:
            if col not in df_results.columns:
                df_results[col] = pd.NA
    else:
        df_results = pd.DataFrame(columns=all_results_columns)

    row_data: dict[str, Any] = {"timestamp": timestamp}
    row_data.update({k: run_config[k] for k in run_config})
    row_data.update({m: history[m][-1] for m in history})
    row_data.update(summary_metrics)
    df_results.loc[len(df_results)] = row_data
    df_results = df_results[all_results_columns]
    df_results.to_excel(all_results_path, index=False)

    logger.info("Writing train predictions...")
    write_predictions(model, train_loader, y_train, dir_name, "train", device)
    logger.info("Writing val predictions...")
    write_predictions(model, val_loader, y_val, dir_name, "val", device)
    logger.info("Writing test predictions...")
    write_predictions(model, test_loader, y_test, dir_name, "test", device)

    return history, x_test, y_test, y_pred_test, model


if __name__ == "__main__":
    setup_logging()
    logger.info("Torch version: %s", torch.__version__)
    logger.info("CUDA available: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    dataset_path = Path(__file__).resolve().parents[3] / "data" / "continuous" / "database_continuous.xlsx"
    df = load_dataset(dataset_path)

    sampler = ParameterSampler(
        HP_DISTRIBUTIONS, n_iter=N_TRIALS, random_state=SEARCH_SEED
    )
    for trial, params in enumerate(sampler, start=1):
        full_params = {**HP_FIXED, **params}
        logger.info("Trial %d/%d - hyperparameters: %s", trial, N_TRIALS, full_params)
        play(df, verbose=False, **full_params)
