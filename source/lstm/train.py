import logging
import os
import sys
from pathlib import Path
from typing import Any

# Prevent TF (loaded transitively via data_preparation) from pre-allocating GPU VRAM.
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_ENABLE_CUDNN_FRONTEND", "0")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared_modules.data_preparation import load_dataset, prepare_data
from shared_modules.logging_setup import setup_logging

logger = logging.getLogger(__name__)


dict_hyperparams: dict[str, Any] = {
    "lookback_window": 25,
    "batch_size": 32,
    "epochs": 50,
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.3,
    "learning_rate": 1e-3,
}


class LSTMClassifier(nn.Module):
    """Sequence classifier that uses a stacked LSTM followed by a linear head.

    The last hidden state of the top LSTM layer is fed into a fully-connected
    layer that produces one logit per class.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.3,
    ) -> None:
        """Initialise the LSTM and the classification head.

        Args:
            input_size: Number of input features at each time step.
            hidden_size: Number of hidden units in each LSTM layer.
            num_layers: Number of stacked LSTM layers.
            num_classes: Number of output classes.
            dropout: Dropout probability applied between LSTM layers (ignored
                when num_layers == 1).
        """
        super().__init__()
        self.lstm = nn.LSTM(
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
            x: Input tensor of shape (batch, sequence_length, input_size).

        Returns:
            Logits tensor of shape (batch, num_classes).
        """
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])


def make_loaders(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Wrap numpy arrays in PyTorch DataLoaders.

    One-hot label arrays are converted to class-index tensors as required by
    CrossEntropyLoss.  Feature arrays stay on CPU; batches are moved to the
    target device inside the training loop.

    Args:
        x_train: Training features, shape (n, window, features).
        x_val: Validation features.
        x_test: Test features.
        y_train: Training labels in one-hot encoding, shape (n, num_classes).
        y_val: Validation labels in one-hot encoding.
        y_test: Test labels in one-hot encoding.
        batch_size: Mini-batch size for all loaders.

    Returns:
        A tuple of (train_loader, val_loader, test_loader).
    """
    def to_dataset(x: np.ndarray, y: np.ndarray) -> TensorDataset:
        return TensorDataset(
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(np.argmax(y, axis=1), dtype=torch.long),
        )

    return (
        DataLoader(to_dataset(x_train, y_train), batch_size=batch_size, shuffle=True),
        DataLoader(to_dataset(x_val, y_val), batch_size=batch_size),
        DataLoader(to_dataset(x_test, y_test), batch_size=batch_size),
    )


def run_epoch(
    model: LSTMClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | str = "cpu",
) -> tuple[float, float]:
    """Run one full pass over a data loader.

    When *optimizer* is provided the model is set to training mode and
    gradients are updated; otherwise the model runs in evaluation mode with
    gradient computation disabled.

    Args:
        model: The LSTM classifier.
        loader: DataLoader yielding (x, y) batches.
        criterion: Loss function (CrossEntropyLoss).
        optimizer: Optimizer for the training pass; ``None`` for evaluation.
        device: Device to move batches to before the forward pass.

    Returns:
        A tuple of (mean_loss, accuracy) over the full loader.
    """
    training = optimizer is not None
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(yb)
    return total_loss / total, correct / total


def plot_history(
    history: dict[str, list[float]],
    dir_name: str,
    save_eps: bool = False,
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
    model: LSTMClassifier,
    loader: DataLoader,
    y_onehot: np.ndarray,
    dir_name: str,
    subset_type: str,
    device: torch.device | str,
) -> None:
    """Write an Excel file with model predictions for a data subset.

    Each row contains the flattened input window, the predicted class index,
    and the ground-truth class index.

    Args:
        model: Trained LSTM classifier.
        loader: DataLoader for the subset (train / val / test).
        y_onehot: One-hot ground-truth labels for the full subset,
            shape (n, num_classes).
        dir_name: Directory where the Excel file is written.
        subset_type: Label used in the output filename (e.g. ``"train"``).
        device: Device to run inference on.
    """
    model.eval()
    all_x, all_preds = [], []
    with torch.no_grad():
        for xb, _ in loader:
            all_preds.append(model(xb.to(device)).argmax(1).cpu().numpy())
            all_x.append(xb.numpy())
    x_arr = np.concatenate(all_x)
    df_out = pd.DataFrame(x_arr.reshape(x_arr.shape[0], -1))
    df_out["predicted"] = np.concatenate(all_preds)
    df_out["y_groundtruth"] = np.argmax(y_onehot, axis=1)
    df_out.to_excel(os.path.join(dir_name, f"{subset_type}_predictions.xlsx"), index=False)


def play(
    df: pd.DataFrame,
    verbose: bool = True,
    save_eps: bool = False,
    **hyperparameters: Any,
) -> tuple[dict[str, list[float]], np.ndarray, np.ndarray, np.ndarray, LSTMClassifier]:
    """Train and evaluate the LSTM classifier.

    Orchestrates data preparation, model construction, the training loop,
    metric logging, plot generation, and prediction export.  Results are
    written to a timestamped sub-directory under ``results/``.

    Args:
        df: Raw dataset DataFrame as returned by ``load_dataset``.
        verbose: When ``True``, log dataset statistics during preparation.
        save_eps: When ``True``, also save EPS versions of all figures.
        **hyperparameters: Training configuration.  Expected keys:
            ``lookback_window``, ``batch_size``, ``epochs``,
            ``hidden_size``, ``num_layers``, ``dropout``,
            ``learning_rate``.

    Returns:
        A tuple of (history, x_test, y_test, y_pred_classes_test, model).
    """
    logger.info("Play function called with hyperparameters: %s", hyperparameters)

    dir_root_results = "results"
    timestamp = str(pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S"))
    dir_name = os.path.join(dir_root_results, f"results_lstm_{timestamp}")
    os.makedirs(dir_name, exist_ok=True)

    with open(os.path.join(dir_name, "hyperparameters.txt"), "wt") as f:
        f.write("Hyperparameters:\n")
        for key, value in hyperparameters.items():
            f.write(f"\t{key}: {value}\n")

    lookback_window: int = hyperparameters["lookback_window"]
    x_train, x_val, x_test, y_train, y_val, y_test, label_encoder = prepare_data(
        df, lookback_window, verbose=verbose
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    train_loader, val_loader, test_loader = make_loaders(
        x_train, x_val, x_test, y_train, y_val, y_test, hyperparameters["batch_size"]
    )

    model = LSTMClassifier(
        input_size=x_train.shape[2],
        hidden_size=hyperparameters["hidden_size"],
        num_layers=hyperparameters["num_layers"],
        num_classes=y_train.shape[1],
        dropout=hyperparameters["dropout"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparameters["learning_rate"])

    history: dict[str, list[float]] = {
        "train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": []
    }
    epochs: int = hyperparameters["epochs"]

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device=device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_acc)
        history["val_accuracy"].append(val_acc)
        logger.info(
            "Epoch %d/%d — loss: %.4f, acc: %.4f, val_loss: %.4f, val_acc: %.4f",
            epoch, epochs, train_loss, train_acc, val_loss, val_acc,
        )

    pd.DataFrame(history).to_csv(os.path.join(dir_name, "training_log.csv"), index=False)
    plot_history(history, dir_name, save_eps=save_eps)

    model.eval()
    all_preds: list[np.ndarray] = []
    all_true: list[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in test_loader:
            all_preds.append(model(xb.to(device)).argmax(1).cpu().numpy())
            all_true.append(yb.numpy())
    y_pred_classes_test = np.concatenate(all_preds)
    y_true_classes_test = np.concatenate(all_true)

    original_labels: np.ndarray = label_encoder.inverse_transform(
        np.arange(len(label_encoder.classes_))
    )
    logger.info("Original Labels: %s", original_labels)

    report = classification_report(
        y_true_classes_test, y_pred_classes_test, target_names=original_labels
    )
    logger.info("Classification Report:\n%s", report)
    with open(os.path.join(dir_name, "classification_report.txt"), "wt") as f:
        f.write("Classification Report:\n")
        f.write(report)

    cm = confusion_matrix(y_true_classes_test, y_pred_classes_test)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, cmap="Reds", fmt="d",
        xticklabels=original_labels, yticklabels=original_labels,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix on Test Data")
    _cm_filename = "confusion_matrix"
    plt.savefig(os.path.join(dir_name, f"{_cm_filename}.png"), bbox_inches="tight")
    if save_eps:
        plt.savefig(os.path.join(dir_name, f"{_cm_filename}.eps"), bbox_inches="tight")
    plt.close()

    all_results_path = os.path.join(dir_root_results, "all_results_lstm.xlsx")
    if os.path.exists(all_results_path):
        df_results = pd.read_excel(all_results_path)
    else:
        df_results = pd.DataFrame(
            columns=["timestamp"] + list(hyperparameters.keys()) + list(history.keys())
        )

    lst_row = (
        [timestamp]
        + [str(hyperparameters[k]) for k in hyperparameters]
        + [str(history[m][-1]) for m in history]
    )
    df_results.loc[len(df_results)] = lst_row
    df_results.to_excel(all_results_path, index=False)

    logger.info("Writing train predictions...")
    write_predictions(model, train_loader, y_train, dir_name, "train", device)
    logger.info("Writing val predictions...")
    write_predictions(model, val_loader, y_val, dir_name, "val", device)
    logger.info("Writing test predictions...")
    write_predictions(model, test_loader, y_test, dir_name, "test", device)

    return history, x_test, y_test, y_pred_classes_test, model


if __name__ == "__main__":
    setup_logging()
    logger.info("Torch version: %s", torch.__version__)
    logger.info("CUDA available: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    dataset_path = Path(__file__).resolve().parents[2] / "data" / "DataBaseTCN.xlsx"
    df = load_dataset(dataset_path)
    play(df, verbose=True, **dict_hyperparams)
