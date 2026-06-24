"""
Collect the best trial (highest val_macro_f1) from every all_results_*.xlsx file
found under results/day_tractor and write the combined table to
results/day_tractor/best_models.xlsx.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared_modules.config import OPTIMIZATION_TRIALS

RESULTS_ROOT = Path(__file__).resolve().parents[2] / "results" / "day_tractor"
OUTPUT_PATH  = RESULTS_ROOT / "best_models.xlsx"


def main() -> None:
    xlsx_files = sorted(RESULTS_ROOT.rglob("all_results_*.xlsx"))
    if not xlsx_files:
        print("No all_results_*.xlsx files found under", RESULTS_ROOT)
        return

    best_rows: list[pd.DataFrame] = []
    for path in xlsx_files:
        df = pd.read_excel(path)
        if len(df) < OPTIMIZATION_TRIALS:
            print(f"Skipping {path.name}: only {len(df)}/{OPTIMIZATION_TRIALS} trials complete")
            continue
        if "val_macro_f1" not in df.columns:
            print(f"Skipping {path.name}: column 'val_macro_f1' not found")
            continue
        best_row = df.loc[[df["val_macro_f1"].idxmax()]]
        best_rows.append(best_row)
        print(f"{path.name}: best val_macro_f1 = {best_row['val_macro_f1'].values[0]:.6f}")

    if not best_rows:
        print("No valid files to aggregate.")
        return

    combined = pd.concat(best_rows, ignore_index=True)
    combined = combined.sort_values("test_macro_f1", ascending=False).reset_index(drop=True)
    combined.to_excel(OUTPUT_PATH, index=False)
    print(f"\nWrote {len(combined)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
