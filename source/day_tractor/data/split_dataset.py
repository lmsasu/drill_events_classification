"""
Splits data/database_day_tractor.xlsx into train / val / test Excel files.

Strategy
--------
* Unit of split: contiguous segment — a maximal run of consecutive rows
  sharing the same (Tractor Brand, Date, Task).
* Segments are distributed per stratum (tractor × task) so that each
  stratum contributes ~TRAIN_RATIO rows to train, ~VAL_RATIO to val, and
  ~TEST_RATIO to test, with at least one segment per stratum per split.
* Both tractor brands and all activity classes appear in every subset.
* No row belongs to more than one subset (no data leakage).

Output files
------------
  data/day_tractor/train_day_tractor.xlsx
  data/day_tractor/val_day_tractor.xlsx
  data/day_tractor/test_day_tractor.xlsx
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared_modules.logging_setup import setup_logging

# ── configuration ────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "day_tractor"
INPUT_FILE = DATA_DIR / "database_day_tractor.xlsx"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6, "Ratios must sum to 1"

RANDOM_SEED = 42
DATE_FORMAT = "%d.%m.%Y"
# ─────────────────────────────────────────────────────────────────────────────


def load_data(path: Path) -> pd.DataFrame:
    """
    Load the dataset from an Excel file and normalise the Date column.

    Args:
        path: Absolute path to the source .xlsx file.

    Returns:
        DataFrame sorted ascending by ID with Date parsed as datetime.
    """
    df = pd.read_excel(path)
    df["Date"] = pd.to_datetime(df["Date"], format=DATE_FORMAT)
    df = df.sort_values("ID").reset_index(drop=True)
    return df


def build_segments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label every row with a contiguous-segment identifier.

    A new segment starts whenever any of Tractor Brand, Date, or Task changes
    relative to the previous row.  The resulting ``seg_id`` column increases
    monotonically and uniquely identifies each maximal homogeneous block.

    Args:
        df: Raw DataFrame sorted by ID, containing columns
            ``Tractor Brand``, ``Date``, and ``Task``.

    Returns:
        The same DataFrame with an additional integer ``seg_id`` column.
    """
    df["seg_id"] = (
        (df["Task"] != df["Task"].shift())
        | (df["Tractor Brand"] != df["Tractor Brand"].shift())
        | (df["Date"] != df["Date"].shift())
    ).cumsum()
    return df


def compute_segment_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a one-row-per-segment summary table used for split assignment.

    Args:
        df: DataFrame produced by :func:`build_segments`, containing
            ``seg_id``, ``Tractor Brand``, ``Date``, ``Task``, and ``ID``.

    Returns:
        DataFrame indexed by ``seg_id`` with columns:
        ``tractor``, ``date``, ``task``, ``n_rows`` (row count of the segment),
        and ``stratum`` (two-letter tractor prefix + ``_`` + task label).
    """
    segs = (
        df.groupby("seg_id")
        .agg(
            tractor=("Tractor Brand", "first"),
            date=("Date", "first"),
            task=("Task", "first"),
            n_rows=("ID", "count"),  # row count drives the proportional split
        )
        .reset_index()
    )
    # Stratum key: two-letter tractor prefix + task (e.g. "La_st", "Jo_pd").
    # Each stratum is split independently to balance both class and tractor
    # representation across train / val / test.
    segs["stratum"] = segs["tractor"].str[:2] + "_" + segs["task"]
    return segs


def assign_splits(
    segs: pd.DataFrame,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> pd.Series:
    """
    Assign each segment to train, val, or test without overlap.

    For each stratum independently: shuffle segments with ``seed``, then use
    cumulative-row boundaries to assign val (~val_ratio of stratum rows),
    test (~test_ratio), and train (remainder).  When a stratum has ≥ 3
    segments, at least one segment is guaranteed in every split.

    Args:
        segs: Segment table from :func:`compute_segment_table`.
        val_ratio: Target fraction of each stratum's rows to place in val.
        test_ratio: Target fraction of each stratum's rows to place in test.
        seed: Integer seed for the random number generator, ensuring
            reproducible assignments.

    Returns:
        Series with the same index as ``segs`` and string values
        ``'train'``, ``'val'``, or ``'test'``.
    """
    rng = np.random.default_rng(seed)
    # Default every segment to train; val/test are overwritten below.
    split_col = pd.Series("train", index=segs.index, dtype=str)

    for _, grp in segs.groupby("stratum"):
        n = len(grp)
        stratum_rows = grp["n_rows"].sum()
        # Row-count targets, not segment-count targets, so that large segments
        # (e.g. long "st" runs) don't distort the final ratio.
        val_target = val_ratio * stratum_rows
        test_target = test_ratio * stratum_rows

        idx = grp.index.tolist()
        perm = rng.permutation(n)  # reproducible shuffle within stratum
        s_idx = [idx[i] for i in perm]
        # Cumulative row count over the shuffled segment order.
        cumsum = np.cumsum(grp.loc[s_idx, "n_rows"].values)

        # First boundary: where cumulative rows first exceed the val target.
        cut1 = int(np.searchsorted(cumsum, val_target, side="right"))
        # Second boundary: where cumulative rows first exceed val + test target.
        cut2 = int(np.searchsorted(cumsum, val_target + test_target, side="right"))

        if n >= 3:
            # Guarantee at least one segment per split when the stratum is
            # large enough; rare strata with < 3 segments go entirely to train.
            cut1 = max(cut1, 1)
            cut2 = max(cut2, cut1 + 1)
            cut2 = min(cut2, n - 1)  # leave at least one segment for train

        for i in s_idx[:cut1]:
            split_col[i] = "val"
        for i in s_idx[cut1:cut2]:
            split_col[i] = "test"
        for i in s_idx[cut2:]:
            split_col[i] = "train"

    return split_col


KEEP_COLS = ["ID", "Tractor Brand", "Date and time", "Dist (m)", "LegSpeed", "LegHead (degrees)", "Task"]


COL_RENAME = {
    "Dist (m)": "Dist",
    "LegSpeed": "Speed",
    "LegHead (degrees)": "Heading",
}


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select, parse, and rename the modelling columns from the full dataset.

    Transformations applied:

    * ``LegSpeed``: numeric value extracted from strings such as ``"3.2 km/h"``
      (everything after the first whitespace is discarded); renamed to ``Speed``.
    * ``Dist (m)``: kept as-is, renamed to ``Dist``.
    * ``LegHead (degrees)``: kept as-is, renamed to ``Heading``.
    * All other selected columns are kept and named as in the source file.

    Args:
        df: Full DataFrame containing at least the columns listed in
            ``KEEP_COLS`` plus ``seg_id`` and ``split`` (added by the
            pipeline before this call).

    Returns:
        DataFrame with columns ``seg_id``, ``split``, ``ID``,
        ``Tractor Brand``, ``Date and time``, ``Dist``, ``Speed``,
        ``Heading``, ``Task``, where ``Speed`` is ``float64``.
    """
    out = df[["seg_id", "split"] + KEEP_COLS].copy()
    # LegSpeed is stored as "3.2 km/h"; split on whitespace and take the
    # first token to discard the unit, then cast to float.
    out["LegSpeed"] = out["LegSpeed"].str.split(expand=True)[0].astype(float)
    out = out.rename(columns=COL_RENAME)
    return out


def verify_splits(df: pd.DataFrame) -> None:
    """
    Assert that the three splits are valid.

    Checks performed:
    * No row ID appears in more than one split.
    * Every activity class (Task) present in the full dataset appears in
      each split.
    * Every tractor present in the initial dataset appears in each split.

    Args:
        df: Full DataFrame with a ``split`` column (values ``'train'``,
            ``'val'``, ``'test'``), ``ID``, ``Task``, and ``Tractor Brand``.

    Raises:
        AssertionError: If any check fails, with a descriptive message.
    """
    # No row may appear in two splits (segment boundaries must be respected).
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        ids_a = set(df.loc[df["split"] == a, "ID"])
        ids_b = set(df.loc[df["split"] == b, "ID"])
        assert ids_a.isdisjoint(ids_b), f"Overlap between {a} and {b}"

    # Every class and every tractor brand must be represented in each split
    # so that evaluation metrics are meaningful across all categories.
    for split in ("train", "val", "test"):
        sub = df[df["split"] == split]
        missing_tasks = set(df["Task"].unique()) - set(sub["Task"].unique())
        missing_tractors = set(df["Tractor Brand"].unique()) - set(
            sub["Tractor Brand"].unique()
        )
        assert not missing_tasks, f"{split} missing tasks: {missing_tasks}"
        assert not missing_tractors, f"{split} missing tractors: {missing_tractors}"


def log_summary(df: pd.DataFrame) -> None:
    """
    Log a compact split summary at INFO level.

    Args:
        df: Full DataFrame with a ``split`` column and ``Tractor Brand``.
    """
    total = len(df)
    logging.info("%-6s  %7s  %6s  %s", "Split", "Rows", "%", "Tractors")
    logging.info("─" * 55)
    for split in ("train", "val", "test"):
        sub = df[df["split"] == split]
        rows = len(sub)
        tractors = sorted(sub["Tractor Brand"].unique())
        logging.info(
            "%-6s  %7s  %5.1f%%  %s", split, f"{rows:,}", rows / total * 100, tractors
        )


def log_distribution_stats(df: pd.DataFrame) -> None:
    """
    Log per-split class and tractor distributions at INFO level.

    For each split the function prints:
    * Total row count and its share of the full dataset.
    * Count and percentage for every ``Task`` class.
    * Count and percentage for every ``Tractor Brand``.

    This helps verify that both class balance and tractor coverage are
    acceptable before training.

    Args:
        df: Full DataFrame with ``split``, ``Task``, and ``Tractor Brand``
            columns (output of :func:`prepare_features` merged with
            ``split``).
    """
    total = len(df)
    for split in ("train", "val", "test"):
        sub = df[df["split"] == split]
        n = len(sub)
        logging.info(
            "#### %s  %s rows  (%.1f%% of total) ####",
            split.upper(), f"{n:,}", n / total * 100,
        )

        logging.info("  Task distribution:")
        task_counts = sub["Task"].value_counts().sort_index()
        for task, count in task_counts.items():
            logging.info("    %-25s  %6s  (%5.1f%%)", task, f"{count:,}", count / n * 100)

        logging.info("  Tractor distribution:")
        tractor_counts = sub["Tractor Brand"].value_counts().sort_index()
        for tractor, count in tractor_counts.items():
            logging.info("    %-25s  %6s  (%5.1f%%)", tractor, f"{count:,}", count / n * 100)


def save_splits(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Write each split to a separate Excel file in ``output_dir``.

    The internal ``seg_id`` and ``split`` columns are dropped before saving.
    Output filenames follow the pattern ``{split}_day_tractor.xlsx``.

    Args:
        df: Full DataFrame with ``seg_id`` and ``split`` columns.
        output_dir: Directory where the three output files are written.
    """
    drop_cols = ["seg_id", "split"]
    for split in ("train", "val", "test"):
        subset = df[df["split"] == split].drop(columns=drop_cols)
        out = output_dir / f"{split}_day_tractor.xlsx"
        subset.to_excel(out, index=False)
        logging.info("Saved %s  (%s rows)", out, f"{len(subset):,}")


def main() -> None:
    """Load, segment, split, verify, and save the dataset."""
    setup_logging()
    logging.info("Loading %s ...", INPUT_FILE)
    df = load_data(INPUT_FILE)
    df = build_segments(df)

    segs = compute_segment_table(df)
    split_col = assign_splits(segs, VAL_RATIO, TEST_RATIO, RANDOM_SEED)
    segs["split"] = split_col

    df = df.merge(segs[["seg_id", "split"]], on="seg_id", how="left")
    df = prepare_features(df)

    verify_splits(df)
    log_summary(df)
    log_distribution_stats(df)
    save_splits(df, DATA_DIR)


if __name__ == "__main__":
    main()
