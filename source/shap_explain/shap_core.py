"""Shared utilities for SHAP-based explanation of sequence classifiers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import torch
import torch.nn as nn

from shared_modules.timing import timeit

logger = logging.getLogger(__name__)

# Supported explainer backends.
EXPLAINER_CHOICES: list[str] = ["deep", "gradient", "kernel"]


# ── Run discovery ──────────────────────────────────────────────────────────────


def find_best_run(
    all_results_path: Path,
    results_dir: Path,
    model_prefix: str,
    rank_by: str = "test_f1",
) -> tuple[Path, dict[str, float]]:
    """Return the run directory and metric dict for the trial with the highest ``rank_by`` score.

    Args:
        all_results_path: Path to the aggregated Excel results file.
        results_dir: Directory that contains all timestamped run subdirectories.
        model_prefix: Prefix of run subdirectory names (e.g. ``"results_gru_"``).
        rank_by: Column in the results file to maximise when selecting the best run.

    Returns:
        ``(run_dir, metrics)`` where ``run_dir`` is the matching subdirectory and
        ``metrics`` is a dict of all ``train_*``, ``val_*``, and ``test_*`` columns
        for that row.
    """
    df = pd.read_excel(all_results_path)
    if rank_by not in df.columns:
        raise ValueError(
            f"Column '{rank_by}' not found in {all_results_path}. "
            f"Available columns: {list(df.columns)}"
        )
    best_idx = df[rank_by].idxmax()
    best_row = df.loc[best_idx]

    raw_ts = best_row["timestamp"]
    # Excel may silently convert "2026-05-23_08-31-55" to a datetime object.
    if isinstance(raw_ts, pd.Timestamp):
        timestamp = raw_ts.strftime("%Y-%m-%d_%H-%M-%S")
    else:
        timestamp = str(raw_ts).strip()

    run_dir = results_dir / f"{model_prefix}{timestamp}"
    if not run_dir.exists():
        raise FileNotFoundError(
            f"Run directory not found: {run_dir}\n"
            f"Timestamp read from Excel: {timestamp!r}"
        )

    metrics = {
        c: float(best_row[c])
        for c in df.columns
        if c.startswith(("train_", "val_", "test_"))
    }
    logger.info(
        "Best run: %s  (%s = %.4f)", run_dir.name, rank_by, float(best_row[rank_by])
    )
    return run_dir, metrics


def load_hyperparameters(run_dir: Path) -> dict:
    """Parse the ``Hyperparameters:`` section of ``hyperparameters.txt``.

    Args:
        run_dir: A timestamped run directory containing ``hyperparameters.txt``.

    Returns:
        Dict mapping parameter names to their values (int/float/str as appropriate).
    """
    hp: dict = {}
    in_section = False
    with open(run_dir / "hyperparameters.txt") as f:
        for line in f:
            stripped = line.strip()
            if stripped == "Hyperparameters:":
                in_section = True
                continue
            if stripped in ("Runtime flags:", "Run summary:"):
                in_section = False
                continue
            if in_section and ": " in stripped:
                key, _, val_str = stripped.partition(": ")
                val_str = val_str.strip()
                try:
                    val: int | float | str = int(val_str)
                except ValueError:
                    try:
                        val = float(val_str)
                    except ValueError:
                        val = val_str
                hp[key.strip()] = val
    return hp


# ── SHAP computation ───────────────────────────────────────────────────────────


def _normalise_shap(raw) -> np.ndarray:
    """Convert any SHAP output format to ``(n_classes, n_samples, seq_len, n_features)``."""
    if isinstance(raw, list):
        # DeepExplainer / GradientExplainer: list of n_classes arrays
        arr = np.stack(raw, axis=0)  # (n_classes, n_samples, seq_len, n_features)
    else:
        arr = np.asarray(raw)
        if arr.ndim == 3:
            # Single-class model or kernel output: (n_samples, seq_len, n_features)
            arr = arr[np.newaxis]
        elif arr.ndim == 4 and arr.shape[0] != arr.shape[-1]:
            # Could be (n_samples, seq_len, n_features, n_classes) — transpose
            if arr.shape[-1] < arr.shape[0]:
                arr = arr.transpose(3, 0, 1, 2)
    return arr  # (n_classes, n_samples, seq_len, n_features)


@timeit
def build_explainer(
    explainer_type: str,
    model: nn.Module,
    background: torch.Tensor,
) -> Any:
    """Instantiate the requested SHAP explainer.

    Args:
        explainer_type: One of ``"deep"``, ``"gradient"``, or ``"kernel"``.
        model: Trained PyTorch model in eval mode; must accept
            ``(batch, seq_len, n_features)`` input.
        background: Background dataset tensor of shape
            ``(n_background, seq_len, n_features)`` on CPU.

    Returns:
        A configured SHAP explainer object.
    """
    explainer_type = explainer_type.lower()
    if explainer_type not in EXPLAINER_CHOICES:
        raise ValueError(
            f"explainer_type must be one of {EXPLAINER_CHOICES}, got {explainer_type!r}"
        )
    model.eval()

    if explainer_type == "deep":
        return shap.DeepExplainer(model, background)

    if explainer_type == "gradient":
        return shap.GradientExplainer(model, background)

    # kernel — model-agnostic; wraps model to accept flattened input
    seq_len, n_feat = background.shape[1], background.shape[2]

    def _model_fn(x_flat: np.ndarray) -> np.ndarray:
        t = torch.tensor(x_flat.reshape(-1, seq_len, n_feat), dtype=torch.float32)
        with torch.no_grad():
            return torch.softmax(model(t), dim=1).numpy()

    bg_flat = background.numpy().reshape(len(background), -1)
    return shap.KernelExplainer(_model_fn, bg_flat)


@timeit
def compute_shap(
    explainer,
    x_explain: torch.Tensor,
    explainer_type: str,
) -> np.ndarray:
    """Run the explainer and return SHAP values shaped ``(n_classes, n_samples, seq_len, n_features)``.

    Args:
        explainer: A SHAP explainer object returned by :func:`build_explainer`.
        x_explain: Tensor of samples to explain, shape ``(n_samples, seq_len, n_features)``.
        explainer_type: Must match what was used to build the explainer.

    Returns:
        SHAP value array of shape ``(n_classes, n_samples, seq_len, n_features)``.
    """
    if explainer_type == "kernel":
        x_in = x_explain.numpy().reshape(len(x_explain), -1)
    else:
        x_in = x_explain
    raw = explainer.shap_values(x_in)
    return _normalise_shap(raw)


# ── Aggregation ────────────────────────────────────────────────────────────────


@timeit
def aggregate_shap(
    sv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Summarise ``(n_classes, n_samples, seq_len, n_features)`` SHAP values.

    Returns:
        ``(sv_signed_2d, sv_abs_2d, sv_heatmap)`` where:

        - ``sv_signed_2d``: signed mean across classes and time →
          ``(n_samples, n_features)``, used for beeswarm summary plot.
        - ``sv_abs_2d``: mean |SHAP| across classes and time →
          ``(n_samples, n_features)``, used for bar chart.
        - ``sv_heatmap``: mean |SHAP| across classes and samples →
          ``(seq_len, n_features)``, used for heatmap.
    """
    sv_signed_2d = sv.mean(axis=0).mean(axis=1)  # (n_samples, n_features)
    sv_abs_2d = np.abs(sv).mean(axis=(0, 2))  # (n_samples, n_features)
    sv_heatmap = np.abs(sv).mean(axis=(0, 1))  # (seq_len, n_features)
    return sv_signed_2d, sv_abs_2d, sv_heatmap


# ── Plots ──────────────────────────────────────────────────────────────────────


def _save(out_dir: Path, stem: str, save_eps: bool) -> None:
    plt.savefig(out_dir / f"{stem}.png", bbox_inches="tight")
    if save_eps:
        plt.savefig(out_dir / f"{stem}.eps", bbox_inches="tight")
    plt.close()


def plot_summary(
    sv_signed_2d: np.ndarray,
    x_2d: np.ndarray,
    feature_names: list[str],
    out_dir: Path,
    save_eps: bool = True,
) -> None:
    """Beeswarm SHAP summary plot (feature value coloured).

    Args:
        sv_signed_2d: Signed SHAP values ``(n_samples, n_features)``.
        x_2d: Feature values ``(n_samples, n_features)`` used for dot colouring.
        feature_names: Names for each feature column.
        out_dir: Output directory.
        save_eps: Also write an EPS copy.
    """
    shap.summary_plot(
        sv_signed_2d,
        x_2d,
        feature_names=feature_names,
        show=False,
        plot_size=(10, max(4, len(feature_names) * 0.35)),
    )
    _save(out_dir, "shap_summary", save_eps)


def plot_bar(
    sv_abs_2d: np.ndarray,
    feature_names: list[str],
    out_dir: Path,
    save_eps: bool = True,
) -> None:
    """Horizontal bar chart of mean |SHAP| per feature (global importance).

    Args:
        sv_abs_2d: Absolute SHAP values ``(n_samples, n_features)``.
        feature_names: Names for each feature column.
        out_dir: Output directory.
        save_eps: Also write an EPS copy.
    """
    mean_abs = sv_abs_2d.mean(axis=0)
    order = np.argsort(mean_abs)  # ascending for horizontal bar (bottom = highest)
    fig, ax = plt.subplots(figsize=(10, max(4, len(feature_names) * 0.4)))
    ax.barh(
        [feature_names[i] for i in order],
        mean_abs[order],
        color="#e84040",
    )
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Feature importance — mean absolute SHAP (all classes & time steps)")
    plt.tight_layout()
    _save(out_dir, "shap_bar", save_eps)


def plot_heatmap(
    sv_heatmap: np.ndarray,
    feature_names: list[str],
    out_dir: Path,
    save_eps: bool = True,
) -> None:
    """Heatmap of mean |SHAP| across time steps × features.

    Args:
        sv_heatmap: ``(seq_len, n_features)`` array averaged over classes and samples.
        feature_names: Names for each feature column.
        out_dir: Output directory.
        save_eps: Also write an EPS copy.
    """
    seq_len = sv_heatmap.shape[0]
    fig, ax = plt.subplots(
        figsize=(max(8, len(feature_names) * 0.6), max(4, seq_len * 0.28))
    )
    yticks = [f"t-{seq_len - i}" for i in range(seq_len)]
    sns.heatmap(
        sv_heatmap,
        ax=ax,
        xticklabels=feature_names,
        yticklabels=yticks,
        cmap="Reds",
        annot=False,
        linewidths=0,
    )
    ax.set_xlabel("Feature")
    ax.set_ylabel("Time step relative to prediction point")
    ax.set_title("SHAP — time step × feature  (mean |SHAP| across samples & classes)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    _save(out_dir, "shap_heatmap", save_eps)


# ── Markdown report ────────────────────────────────────────────────────────────


@timeit
def write_report(
    model_name: str,
    run_dir: Path,
    hp: dict,
    metrics: dict[str, float],
    sv_abs_2d: np.ndarray,
    feature_names: list[str],
    explainer_type: str,
    n_background: int,
    n_explain: int,
    out_dir: Path,
) -> None:
    """Write a Markdown SHAP analysis report to ``shap_report.md``.

    Args:
        model_name: Display name for the model (e.g. ``"GRU"``).
        run_dir: The run directory whose checkpoint was explained.
        hp: Hyperparameter dict as returned by :func:`load_hyperparameters`.
        metrics: Metric dict (``train_*``, ``val_*``, ``test_*`` keys).
        sv_abs_2d: Absolute SHAP values ``(n_samples, n_features)``.
        feature_names: Feature column names.
        explainer_type: Explainer backend used.
        n_background: Number of background samples.
        n_explain: Number of test samples explained.
        out_dir: Directory where the report is written.
    """
    mean_abs = sv_abs_2d.mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    top_n = min(15, len(feature_names))

    lines: list[str] = [
        f"# SHAP Analysis Report — {model_name}",
        "",
        f"**Source run:** `{run_dir.name}`",
        f"**Explainer backend:** {explainer_type}",
        f"**Background samples (training set):** {n_background}",
        f"**Explained samples (test set):** {n_explain}",
        "",
        "## Hyperparameters",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
    ]
    for k, v in hp.items():
        lines.append(f"| `{k}` | {v} |")

    lines += [
        "",
        "## Model Performance on Test Set",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ]
    for k, v in sorted(metrics.items()):
        if k.startswith("test_"):
            lines.append(f"| {k.replace('test_', '')} | {v:.4f} |")

    lines += [
        "",
        f"## Global Feature Importance (Top {top_n})",
        "",
        "Mean absolute SHAP value aggregated across all explained test samples, "
        "all time steps in the lookback window, and all output classes.",
        "",
        "| Rank | Feature | Mean \\|SHAP\\| |",
        "|------|---------|--------------|",
    ]
    for rank, idx in enumerate(order[:top_n], start=1):
        lines.append(f"| {rank} | `{feature_names[idx]}` | {mean_abs[idx]:.6f} |")

    lines += [
        "",
        "## Output Files",
        "",
        "| File | Description |",
        "|------|-------------|",
        "| `shap_summary.png` | Beeswarm plot — distribution of SHAP values per feature across "
        "test samples; dot colour encodes feature value (red = high, blue = low). |",
        "| `shap_bar.png` | Bar chart — mean \\|SHAP\\| per feature, global importance ranking. |",
        "| `shap_heatmap.png` | Heatmap — mean \\|SHAP\\| per time step × feature, "
        "revealing which features matter most at which position in the lookback window. |",
        "",
        "## Interpretation Notes",
        "",
        "- SHAP values are computed per output class using the explainer's class-wise "
        "decomposition. Importance scores reported here are averaged in absolute value "
        "across all classes to give a class-agnostic feature ranking.",
        "- The heatmap rows correspond to consecutive time steps inside the lookback "
        f"window of length **{hp.get('lookback_window', '?')}**. "
        "Row `t-1` is the most recent observation immediately before the predicted "
        "point; row `t-N` is the oldest.",
        "- Features concentrated near `t-1` (bottom rows of the heatmap) are primarily "
        "driven by recent dynamics; features spread uniformly across rows carry "
        "persistent long-range information.",
        "- The beeswarm plot preserves the sign of SHAP values: a cluster of red dots "
        "shifted right means high feature values push the model toward a particular class.",
    ]

    report_path = out_dir / "shap_report.md"
    report_path.write_text("\n".join(lines) + "\n")
    logger.info("Report written to %s", report_path)
