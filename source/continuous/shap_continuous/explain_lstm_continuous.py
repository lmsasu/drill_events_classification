"""SHAP explanation script for the LSTM classifier.

Selects the best LSTM run by test F1, reloads its checkpoint, computes SHAP
values on the test set, and writes plots + a Markdown report to
``results/SHAP_LSTM/``.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from continuous.lstm_continuous.train_lstm import LSTMClassifier
from continuous.data.data_preparation import load_dataset, prepare_data
from shared_modules.logging_setup import setup_logging
from continuous.shap_continuous.shap_core import (
    EXPLAINER_CHOICES,
    aggregate_shap,
    build_explainer,
    compute_shap,
    find_best_run,
    load_hyperparameters,
    plot_bar,
    plot_heatmap,
    plot_summary,
    write_report,
)

logger = logging.getLogger(__name__)


# -- Configuration --------------------------------------------------------------
# Explainer backend - one of: "deep", "gradient", "kernel"
# Use "gradient" for GRU/LSTM: DeepExplainer (DeepLIFT) does not support recurrent
# layers and will fail the additivity check. GradientExplainer uses backprop and
# works correctly with all PyTorch RNN modules.
EXPLAINER_TYPE: str = "gradient"

# Number of training samples used as the SHAP background distribution.
N_BACKGROUND: int = 100

# Number of test samples to explain. Set to None to explain the entire test set.
N_EXPLAIN: int | None = 200

# Metric column from all_results_lstm.xlsx used to rank and select the best run.
RANK_BY: str = "test_f1"

SAVE_EPS: bool = True
# ------------------------------------------------------------------------------


def _build_model(hp: dict, input_size: int, num_classes: int) -> LSTMClassifier:
    return LSTMClassifier(
        input_size=input_size,
        hidden_size=int(hp["hidden_size"]),
        num_layers=int(hp["num_layers"]),
        num_classes=num_classes,
        dropout=float(hp["dropout"]),
    )


if __name__ == "__main__":
    setup_logging()

    repo_root = Path(__file__).resolve().parents[3]
    results_dir = repo_root / "results"
    all_results_path = results_dir / "all_results_lstm.xlsx"
    out_dir = results_dir / "SHAP_LSTM"
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = repo_root / "data" / "continuous" / "database_continuous.xlsx"
    df_raw = load_dataset(dataset_path)
    feature_names = [c for c in df_raw.columns if c != "Task"]

    run_dir, metrics = find_best_run(
        all_results_path, results_dir, "results_lstm_", RANK_BY
    )
    hp = load_hyperparameters(run_dir)
    logger.info("Hyperparameters: %s", hp)

    lookback_window = int(hp["lookback_window"])
    x_train, _, x_test, y_train, _, _, label_encoder = prepare_data(
        df_raw, lookback_window, verbose=False
    )
    num_classes = y_train.shape[1]
    input_size = x_train.shape[2]

    model = _build_model(hp, input_size, num_classes)
    state_dict = torch.load(run_dir / "best_model.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    logger.info("Model loaded from %s", run_dir / "best_model.pt")

    rng = np.random.default_rng(42)
    bg_idx = rng.choice(
        len(x_train), size=min(N_BACKGROUND, len(x_train)), replace=False
    )
    background = torch.tensor(x_train[bg_idx], dtype=torch.float32)

    n_explain = min(N_EXPLAIN, len(x_test)) if N_EXPLAIN is not None else len(x_test)
    ex_idx = rng.choice(len(x_test), size=n_explain, replace=False)
    x_explain_np = x_test[ex_idx]
    x_explain = torch.tensor(x_explain_np, dtype=torch.float32)

    logger.info(
        "Building %s explainer (background=%d, explain=%d)...",
        EXPLAINER_TYPE,
        len(background),
        len(x_explain),
    )
    explainer = build_explainer(EXPLAINER_TYPE, model, background)
    sv = compute_shap(explainer, x_explain, EXPLAINER_TYPE)
    logger.info(
        "SHAP array shape (n_classes, n_samples, seq_len, n_features): %s", sv.shape
    )

    sv_signed_2d, sv_abs_2d, sv_heatmap = aggregate_shap(sv)

    x_2d = x_explain_np.mean(axis=1)

    logger.info("Generating plots...")
    plot_summary(sv_signed_2d, x_2d, feature_names, out_dir, SAVE_EPS)
    plot_bar(sv_abs_2d, feature_names, out_dir, SAVE_EPS)
    plot_heatmap(sv_heatmap, feature_names, out_dir, SAVE_EPS)

    write_report(
        model_name="LSTM",
        run_dir=run_dir,
        hp=hp,
        metrics=metrics,
        sv_abs_2d=sv_abs_2d,
        feature_names=feature_names,
        explainer_type=EXPLAINER_TYPE,
        n_background=len(background),
        n_explain=len(x_explain),
        out_dir=out_dir,
    )
    logger.info("SHAP analysis complete. Results saved to %s", out_dir)
