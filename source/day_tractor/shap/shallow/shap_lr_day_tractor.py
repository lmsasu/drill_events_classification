"""
SHAP feature-importance analysis for the best Logistic Regression trial.

Explainer: shap.LinearExplainer applied to the fitted LogisticRegression step.
           The StandardScaler is applied first; SHAP values are therefore in
           terms of standardised features.  Feature ranking (which of the three
           signals matters most) is unaffected by the standardisation.
Retraining: train + val combined.  See shap_shallow_common for rationale.
"""

import sys
from pathlib import Path

import shap
from sklearn.linear_model import LogisticRegression
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

MODEL_NAME  = "LR"
RESULTS_DIR = RESULTS_ROOT / "shallow" / MODEL_NAME
OUTPUT_DIR  = RESULTS_ROOT / "shap" / MODEL_NAME
ALL_RESULTS = RESULTS_DIR  / "all_results_lr.xlsx"


def main() -> None:
    row         = load_best_row(ALL_RESULTS)
    window_size = int(row["window_size"])
    print(f"Best trial  val_macro_f1={row['val_macro_f1']:.6f}  window_size={window_size}")

    X_train, y_train, X_val, y_val, X_test, y_test = load_flat_arrays(window_size)
    X_tv, y_tv = combine_train_val(X_train, y_train, X_val, y_val)
    print(f"Retraining on {len(X_tv)} train+val samples, evaluating SHAP on {len(X_test)} test samples.")

    class_weight = "balanced" if row.get("use_class_weights", True) else None
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            C            = float(row["C"]),
            l1_ratio     = float(row["l1_ratio"]),
            solver       = str(row.get("solver", "saga")),
            max_iter     = int(row.get("max_iter", 3000)),
            class_weight = class_weight,
            random_state = int(row.get("seed", 42)),
        )),
    ])
    pipeline.fit(X_tv, y_tv)

    scaler = pipeline.named_steps["scaler"]
    logreg = pipeline.named_steps["logreg"]
    X_tv_scaled   = scaler.transform(X_tv)
    X_test_scaled = scaler.transform(X_test)

    explainer   = shap.LinearExplainer(logreg, X_tv_scaled)
    shap_values = explainer.shap_values(X_test_scaled)

    importance = aggregate_shap(shap_values, window_size)
    save_results(importance, MODEL_NAME, window_size, OUTPUT_DIR)


if __name__ == "__main__":
    main()
