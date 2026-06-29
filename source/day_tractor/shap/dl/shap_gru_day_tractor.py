"""
SHAP feature-importance analysis for the best GRU trial.

Explainer: shap.GradientExplainer (expected gradients, PyTorch-native).
Weights:   loaded from best_model.pt — no retraining required.
Background: N_BACKGROUND samples drawn randomly from train+val tensors.
"""

import sys
from pathlib import Path

import numpy as np
import shap
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from day_tractor.dl.train_gru_day_tractor import GRUClassifier
from day_tractor.shap.dl.shap_dl_common import (
    RESULTS_ROOT,
    aggregate_shap_dl,
    load_best_row,
    load_sequence_tensors,
    save_results,
)

MODEL_NAME     = "GRU"
RESULTS_DIR    = RESULTS_ROOT / "dl" / MODEL_NAME
OUTPUT_DIR     = RESULTS_ROOT / "shap" / MODEL_NAME
ALL_RESULTS    = RESULTS_DIR  / "all_results_gru.xlsx"
N_BACKGROUND   = 100
N_TEST_SAMPLES = 300


def main() -> None:
    row         = load_best_row(ALL_RESULTS)
    window_size = int(row["window_size"])
    timestamp   = str(row["timestamp"])
    print(f"Best trial  val_macro_f1={row['val_macro_f1']:.6f}  window_size={window_size}")

    X_train, y_train, X_val, y_val, X_test, y_test, n_classes = load_sequence_tensors(window_size)
    X_tv = torch.cat([X_train, X_val], dim=0)
    print(f"Train+val: {len(X_tv)} sequences.  Test: {len(X_test)} sequences.")

    model = GRUClassifier(
        input_size  = 3,
        hidden_size = int(row["hidden_size"]),
        num_layers  = int(row["num_layers"]),
        num_classes = n_classes,
        dropout     = float(row["dropout"]),
    )
    ckpt = RESULTS_DIR / f"results_gru_{timestamp}" / "best_model.pt"
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    print(f"Loaded checkpoint: {ckpt}")

    rng        = np.random.default_rng(42)
    bg_idx     = rng.choice(len(X_tv),   size=min(N_BACKGROUND,   len(X_tv)),   replace=False)
    test_idx   = rng.choice(len(X_test), size=min(N_TEST_SAMPLES, len(X_test)), replace=False)
    X_bg       = X_tv[bg_idx]
    X_test_sub = X_test[test_idx]
    print(f"GradientExplainer: background={len(X_bg)}, explaining {len(X_test_sub)} test samples.")

    explainer   = shap.GradientExplainer(model, X_bg)
    shap_values = explainer.shap_values(X_test_sub)

    importance = aggregate_shap_dl(shap_values)
    save_results(importance, MODEL_NAME, window_size, OUTPUT_DIR)


if __name__ == "__main__":
    main()
