"""
Shared utilities for SHAP feature-importance analysis of day-tractor DL models.

Model loading
-------------
DL model weights are loaded directly from the best_model.pt checkpoint saved
during the original training run — no retraining is required.

SHAP explainer
--------------
shap.GradientExplainer is used (expected gradients, PyTorch-native).  It
supports GRU, LSTM, and TCN because PyTorch autograd can differentiate through
all three architectures.  Background samples are drawn from the combined
train+val tensors.

SHAP values for a multiclass model are returned as a list of n_classes arrays,
each of shape (n_samples, window_size, n_features) — already in the natural 3D
shape, so no reshaping is needed.

Normalization
-------------
Mean absolute SHAP is computed across samples and time steps, yielding a (3,)
importance vector for [Dist, Speed, Heading].  Averaging over time steps
normalises for window_size so that models with different window sizes are
directly comparable.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from day_tractor.data.dataset import TractorActivityDataset

DATA_DIR      = Path(__file__).resolve().parents[4] / "data"    / "day_tractor"
RESULTS_ROOT  = Path(__file__).resolve().parents[4] / "results" / "day_tractor"
FEATURE_NAMES = ["Dist", "Speed", "Heading"]
N_FEATURES    = len(FEATURE_NAMES)


def load_best_row(all_results_path: Path) -> pd.Series:
    df = pd.read_excel(all_results_path)
    return df.loc[df["val_macro_f1"].idxmax()]


def load_sequence_tensors(
    window_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Load train/val/test as float32 tensors of shape (n, window_size, 3).

    Returns (X_train, y_train, X_val, y_val, X_test, y_test, n_classes).
    """
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

    def _t(ds: TractorActivityDataset):
        return (
            torch.tensor(ds.windows, dtype=torch.float32),
            torch.tensor(ds.labels,  dtype=torch.long),
        )

    X_train, y_train = _t(train_ds)
    X_val,   y_val   = _t(val_ds)
    X_test,  y_test  = _t(test_ds)
    return X_train, y_train, X_val, y_val, X_test, y_test, train_ds.n_classes


def aggregate_shap_dl(
    shap_values: list[np.ndarray] | np.ndarray,
) -> np.ndarray:
    """Convert raw SHAP output to a normalised per-feature importance vector.

    Parameters
    ----------
    shap_values:
        List of n_classes arrays each (n_samples, window_size, n_features),
        as returned by GradientExplainer for multiclass models; or a single
        array of the same shape, or a 4D array with a class axis.

    Returns
    -------
    importance : np.ndarray, shape (N_FEATURES,)
        Mean |SHAP| per feature, averaged across samples, time steps, and
        classes.  Averaging over time steps normalises for window_size.
    """
    if isinstance(shap_values, list):
        # GradientExplainer multiclass: list of n_classes arrays (n, T, F)
        abs_shap = np.stack([np.abs(sv) for sv in shap_values]).mean(axis=0)
    else:
        arr = np.abs(shap_values)
        if arr.ndim == 4:
            # (n_samples, window_size, n_features, n_classes) — mean over class axis
            abs_shap = arr.mean(axis=-1)
        else:
            abs_shap = arr  # (n_samples, window_size, n_features)

    return abs_shap.mean(axis=(0, 1))  # (N_FEATURES,)


def save_results(
    importance: np.ndarray,
    model_name: str,
    window_size: int,
    output_dir: Path,
) -> None:
    """Save bar-chart (PNG + EPS), raw importance array (.npy), and append to shap_analysis.xlsx."""
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / f"shap_importance_{model_name.lower()}"

    fig, ax = plt.subplots(figsize=(6, 4))
    colours = ["#4C72B0", "#DD8452", "#55A868"]
    bars    = ax.bar(FEATURE_NAMES, importance, color=colours)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Mean |SHAP| (normalised by window size)")
    ax.set_title(f"SHAP Feature Importance — {model_name}  (window={window_size})")
    for bar, val in zip(bars, importance):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.02,
            f"{val:.5f}",
            ha="center", va="bottom", fontsize=9,
        )
    ax.set_ylim(0, max(importance) * 1.18)
    plt.tight_layout()
    plt.savefig(base.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.savefig(base.with_suffix(".eps"),           bbox_inches="tight")
    plt.close()

    np.save(base.with_suffix(".npy"), importance)

    # Append / update row in shared summary spreadsheet
    xlsx_path = output_dir.parent / "shap_analysis.xlsx"
    new_row   = pd.DataFrame([{"model": model_name, **dict(zip(FEATURE_NAMES, importance))}])
    if xlsx_path.exists():
        df = pd.read_excel(xlsx_path)
        df = df[df["model"] != model_name]   # replace existing row for this model
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        df = new_row
    df.to_excel(xlsx_path, index=False)

    print(f"\nSaved → {base}.[png|eps|npy]")
    print(f"Updated → {xlsx_path}")
    for name, val in zip(FEATURE_NAMES, importance):
        print(f"  {name}: {val:.6f}")
