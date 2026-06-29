"""
Shared utilities for SHAP feature-importance analysis of day-tractor shallow models.

Retraining policy
-----------------
The best hyperparameter configuration is identified by the highest val_macro_f1 in
the model's all_results_*.xlsx.  The model is then **retrained on the union of the
training and validation splits (train + val)**.  The validation split was used only
for model *selection* across 100 random-search trials — it was never used to update
any model weights or coefficients within a trial — so including it in the final
training set is legitimate and gives a slightly larger effective training set.
SHAP values are computed on the held-out test split.

Normalization
-------------
The shallow models receive flattened windows of shape (window_size * 3,).
SHAP assigns an importance value to each of the window_size * 3 input dimensions.
After computing SHAP values, this module reshapes them back to
(n_samples, window_size, 3) and takes the **mean** of absolute values across both
samples and time steps, yielding a (3,) importance vector for [Dist, Speed, Heading].
Using the mean (rather than the sum) over time steps implicitly normalises for
window_size: models with larger windows do not accumulate disproportionate SHAP mass.
For multiclass explainers (which return one SHAP array per class) the mean is also
taken across classes before aggregating over time.
"""

import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from day_tractor.data.dataset import TractorActivityDataset

DATA_DIR      = Path(__file__).resolve().parents[4] / "data"    / "day_tractor"
RESULTS_ROOT  = Path(__file__).resolve().parents[4] / "results" / "day_tractor"
FEATURE_NAMES = ["Dist", "Speed", "Heading"]
N_FEATURES    = len(FEATURE_NAMES)


# ── type-safe readers for nullable Excel cells ────────────────────────────────

def _is_na(val: Any) -> bool:
    if val is None:
        return True
    try:
        return bool(pd.isna(val))
    except (TypeError, ValueError):
        return False


def nullable_int(val: Any) -> int | None:
    return None if _is_na(val) else int(val)


def nullable_str(val: Any) -> str | None:
    return None if _is_na(val) else str(val)


# ── best-trial lookup ─────────────────────────────────────────────────────────

def load_best_row(all_results_path: Path) -> pd.Series:
    """Return the row with the highest val_macro_f1."""
    df = pd.read_excel(all_results_path)
    return df.loc[df["val_macro_f1"].idxmax()]


# ── data loading ──────────────────────────────────────────────────────────────

def load_flat_arrays(
    window_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and flatten train / val / test arrays for the given window_size.

    Returns (X_train, y_train, X_val, y_val, X_test, y_test).
    X arrays have shape (n_windows, window_size * N_FEATURES).
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

    def _flat(ds: TractorActivityDataset) -> tuple[np.ndarray, np.ndarray]:
        return ds.windows.reshape(len(ds.windows), -1), ds.labels

    X_train, y_train = _flat(train_ds)
    X_val,   y_val   = _flat(val_ds)
    X_test,  y_test  = _flat(test_ds)
    return X_train, y_train, X_val, y_val, X_test, y_test


def combine_train_val(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return np.concatenate([X_train, X_val]), np.concatenate([y_train, y_val])


# ── SHAP aggregation ──────────────────────────────────────────────────────────

def aggregate_shap(
    shap_values: list[np.ndarray] | np.ndarray,
    window_size: int,
) -> np.ndarray:
    """Convert raw SHAP output to a normalised per-feature importance vector.

    Parameters
    ----------
    shap_values:
        Either a list of (n_samples, flat_features) arrays — one per class,
        as returned by multiclass TreeExplainer / LinearExplainer /
        KernelExplainer — or a single (n_samples, flat_features) array.
    window_size:
        Number of time steps in the flattened window.

    Returns
    -------
    importance : np.ndarray, shape (N_FEATURES,)
        Mean |SHAP| per original feature, averaged across samples, time steps,
        and classes.  The mean over time steps normalises for window_size.
    """
    if isinstance(shap_values, list):
        # older SHAP: list of n_classes arrays each (n_samples, flat_features)
        abs_shap = np.stack([np.abs(sv) for sv in shap_values]).mean(axis=0)
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # newer SHAP KernelExplainer: (n_samples, flat_features, n_classes)
        abs_shap = np.abs(shap_values).mean(axis=-1)
    else:
        abs_shap = np.abs(shap_values)                    # (n_samples, flat)

    shap_3d = abs_shap.reshape(abs_shap.shape[0], window_size, N_FEATURES)
    return shap_3d.mean(axis=(0, 1))                      # (N_FEATURES,)


# ── plotting and saving ───────────────────────────────────────────────────────

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
