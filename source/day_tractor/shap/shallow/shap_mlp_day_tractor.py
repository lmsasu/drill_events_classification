"""
SHAP feature-importance analysis for the best MLP trial.

Explainer: shap.KernelExplainer (model-agnostic) applied to the full pipeline's
           predict_proba.  A small background set and a capped test subsample
           are used to keep runtime feasible.
Retraining: train + val combined.  See shap_shallow_common for rationale.
"""

import ast
import sys
from pathlib import Path

import numpy as np
import shap
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from day_tractor.shap.shallow.shap_shallow_common import (
    RESULTS_ROOT,
    aggregate_shap,
    combine_train_val,
    load_best_row,
    load_flat_arrays,
    save_results,
)

MODEL_NAME     = "MLP"
RESULTS_DIR    = RESULTS_ROOT / "shallow" / MODEL_NAME
OUTPUT_DIR     = RESULTS_ROOT / "shap" / MODEL_NAME
ALL_RESULTS    = RESULTS_DIR  / "all_results_mlp.xlsx"
N_BACKGROUND   = 100
N_TEST_SAMPLES = 300


def _parse_hidden_layer_sizes(val: object) -> tuple[int, ...]:
    """Parse hidden_layer_sizes stored as a string in the xlsx, e.g. '(30, 30)'."""
    if isinstance(val, tuple):
        return val
    parsed = ast.literal_eval(str(val))
    return tuple(parsed) if isinstance(parsed, (list, tuple)) else (int(parsed),)


def main() -> None:
    row         = load_best_row(ALL_RESULTS)
    window_size = int(row["window_size"])
    print(f"Best trial  val_macro_f1={row['val_macro_f1']:.6f}  window_size={window_size}")

    X_train, y_train, X_val, y_val, X_test, y_test = load_flat_arrays(window_size)
    X_tv, y_tv = combine_train_val(X_train, y_train, X_val, y_val)
    print(f"Retraining on {len(X_tv)} train+val samples.")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes = _parse_hidden_layer_sizes(row.get("hidden_layer_sizes", (50,))),
            activation         = str(row.get("activation", "relu")),
            alpha              = float(row.get("alpha", 1e-4)),
            learning_rate_init = float(row.get("learning_rate_init", 1e-3)),
            learning_rate      = str(row.get("learning_rate", "constant")),
            max_iter           = int(row.get("max_iter", 1000)),
            early_stopping     = bool(row.get("early_stopping", False)),
            random_state       = int(row.get("seed", 42)),
        )),
    ])
    pipeline.fit(X_tv, y_tv)

    rng        = np.random.default_rng(42)
    bg_idx     = rng.choice(len(X_tv),   size=min(N_BACKGROUND,   len(X_tv)),   replace=False)
    test_idx   = rng.choice(len(X_test), size=min(N_TEST_SAMPLES, len(X_test)), replace=False)
    X_bg       = X_tv[bg_idx]
    X_test_sub = X_test[test_idx]
    print(f"KernelExplainer: background={len(X_bg)}, explaining {len(X_test_sub)} test samples.")

    explainer   = shap.KernelExplainer(pipeline.predict_proba, X_bg)
    shap_values = explainer.shap_values(X_test_sub, silent=True)

    importance = aggregate_shap(shap_values, window_size)
    save_results(importance, MODEL_NAME, window_size, OUTPUT_DIR)


if __name__ == "__main__":
    main()
