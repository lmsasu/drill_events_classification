"""
SHAP feature-importance analysis for the best Random Forest trial.

Explainer: shap.TreeExplainer  (exact, fast — no subsampling needed).
Retraining: train + val combined.  See shap_shallow_common for rationale.
"""

import sys
from pathlib import Path

import shap
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from day_tractor.shap.shallow.shap_shallow_common import (
    RESULTS_ROOT,
    aggregate_shap,
    combine_train_val,
    load_best_row,
    load_flat_arrays,
    nullable_int,
    nullable_str,
    save_results,
)

MODEL_NAME  = "RF"
RESULTS_DIR = RESULTS_ROOT / "shallow" / MODEL_NAME
OUTPUT_DIR  = RESULTS_ROOT / "shap" / MODEL_NAME
ALL_RESULTS = RESULTS_DIR  / "all_results_rf.xlsx"


def main() -> None:
    row         = load_best_row(ALL_RESULTS)
    window_size = int(row["window_size"])
    print(f"Best trial  val_macro_f1={row['val_macro_f1']:.6f}  window_size={window_size}")

    X_train, y_train, X_val, y_val, X_test, y_test = load_flat_arrays(window_size)
    X_tv, y_tv = combine_train_val(X_train, y_train, X_val, y_val)
    print(f"Retraining on {len(X_tv)} train+val samples, evaluating SHAP on {len(X_test)} test samples.")

    class_weight = "balanced" if row.get("use_class_weights", True) else None
    model = RandomForestClassifier(
        n_estimators     = int(row.get("n_estimators", 100)),
        max_depth        = nullable_int(row.get("max_depth")),
        min_samples_split= int(row.get("min_samples_split", 2)),
        min_samples_leaf = int(row.get("min_samples_leaf", 1)),
        max_features     = nullable_str(row.get("max_features")) or "sqrt",
        criterion        = str(row.get("criterion", "gini")),
        bootstrap        = bool(row.get("bootstrap", True)),
        class_weight     = class_weight,
        random_state     = int(row.get("seed", 42)),
        n_jobs           = int(row.get("n_jobs", 6)),
    )
    model.fit(X_tv, y_tv)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test, check_additivity=False)

    importance = aggregate_shap(shap_values, window_size)
    save_results(importance, MODEL_NAME, window_size, OUTPUT_DIR)


if __name__ == "__main__":
    main()
