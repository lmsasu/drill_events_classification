"""
SHAP feature-importance analysis for the best KNN trial.

Explainer: shap.KernelExplainer (model-agnostic) applied to the full pipeline's
           predict_proba.  KernelExplainer is slow (O(n_background * n_features)
           per sample), so a small background set and a capped test subsample
           are used.
Retraining: train + val combined.  See shap_shallow_common for rationale.
"""

import sys
from pathlib import Path

import numpy as np
import shap
from sklearn.neighbors import KNeighborsClassifier
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

MODEL_NAME       = "KNN"
RESULTS_DIR      = RESULTS_ROOT / "shallow" / MODEL_NAME
OUTPUT_DIR       = RESULTS_ROOT / "shap" / MODEL_NAME
ALL_RESULTS      = RESULTS_DIR  / "all_results_knn.xlsx"
N_BACKGROUND     = 100   # background samples for KernelExplainer
N_TEST_SAMPLES   = 300   # test samples to explain (capped for runtime)


def main() -> None:
    row         = load_best_row(ALL_RESULTS)
    window_size = int(row["window_size"])
    print(f"Best trial  val_macro_f1={row['val_macro_f1']:.6f}  window_size={window_size}")

    X_train, y_train, X_val, y_val, X_test, y_test = load_flat_arrays(window_size)
    X_tv, y_tv = combine_train_val(X_train, y_train, X_val, y_val)
    print(f"Retraining on {len(X_tv)} train+val samples.")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(
            n_neighbors = int(row.get("n_neighbors", 5)),
            p           = int(row.get("p", 2)),
            n_jobs      = int(row.get("n_jobs", 6)),
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
