"""
PyTorch Dataset for the day-tractor activity classification task.

The three split files (train / val / test) produced by ``split_dataset.py``
are each served by one ``TractorActivityDataset`` instance.  Sliding windows
are extracted strictly within contiguous same-task segments so that no window
ever spans a segment boundary (which would mix activity classes and introduce
leakage between splits).

Typical usage
-------------
::

    from sklearn.preprocessing import LabelEncoder
    from torch.utils.data import DataLoader

    train_ds = TractorActivityDataset(
        file_path   = DATA_DIR / "train_day_tractor.xlsx",
        window_size = 16,
    )
    val_ds = TractorActivityDataset(
        file_path     = DATA_DIR / "val_day_tractor.xlsx",
        window_size   = 16,
        label_encoder = train_ds.label_encoder,   # share the fitted encoder
    )
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

FEATURE_COLS = ["Dist", "Speed", "Heading"]
LABEL_COL    = "Task"


class TractorActivityDataset(Dataset):
    """Sliding-window dataset over pre-split tractor activity files.

    Each item is a pair ``(x, y)`` where:

    * ``x`` is a ``float32`` tensor of shape ``(window_size, 3)`` containing
      the feature sequence ``[Dist, Speed, Heading]``.
    * ``y`` is a ``long`` scalar tensor holding the integer class index, suited
      for :class:`torch.nn.CrossEntropyLoss`.

    Windows are built within each contiguous same-task segment; segments
    shorter than ``window_size`` are silently skipped.

    Args:
        file_path: Path to one of the Excel split files produced by
            ``split_dataset.py`` (train, val, or test).
        window_size: Number of consecutive time steps per input window.
        label_encoder: A pre-fitted :class:`~sklearn.preprocessing.LabelEncoder`
            to use for label encoding.  If ``None``, a new encoder is fitted on
            the classes present in this file.  Pass the train dataset's encoder
            to val and test so that class indices are consistent across splits.
    """

    def __init__(
        self,
        file_path: Path,
        window_size: int,
        label_encoder: LabelEncoder | None = None,
    ) -> None:
        self.file_path   = Path(file_path)
        self.window_size = window_size

        df = self._load(self.file_path)
        df = self._rebuild_segments(df)

        self.label_encoder = self._fit_or_reuse_encoder(df, label_encoder)
        self.windows, self.labels = self._build_windows(df)

        logger.info(
            "%s: %d windows from %d segments  (window_size=%d, classes=%s)",
            self.file_path.name,
            len(self.windows),
            df["_seg_id"].nunique(),
            window_size,
            list(self.label_encoder.classes_),
        )

    @property
    def n_features(self) -> int:
        """Number of input features per time step (always 3: Dist, Speed, Heading)."""
        return len(FEATURE_COLS)

    @property
    def n_classes(self) -> int:
        """Number of activity classes known to the label encoder."""
        return len(self.label_encoder.classes_)

    @property
    def classes(self) -> np.ndarray:
        """Ordered array of class name strings."""
        return self.label_encoder.classes_

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights for :class:`torch.nn.CrossEntropyLoss`.

        Returns a ``float32`` tensor of shape ``(n_classes,)`` where each
        weight is ``total_windows / (n_classes * count_of_that_class)``.
        Classes that produced **zero windows** (all their segments are shorter
        than ``window_size``) receive weight ``1.0`` instead of ``0.0``.
        A weight of ``0.0`` would cause ``CrossEntropyLoss`` to divide by zero
        whenever a batch contains only samples from the absent class.

        Returns:
            Tensor of shape ``(n_classes,)`` on CPU.
        """
        counts  = np.bincount(self.labels, minlength=self.n_classes).astype(np.float32)
        total   = counts.sum()
        # Start from 1.0 so absent classes get a neutral weight instead of 0.
        weights = np.ones(self.n_classes, dtype=np.float32)
        present = counts > 0
        weights[present] = total / (self.n_classes * counts[present])
        missing = [self.classes[i] for i, c in enumerate(counts) if c == 0]
        if missing:
            logger.warning(
                "%s: classes with 0 training windows (all segments shorter than "
                "window_size=%d): %s — their loss weight is set to 1.0",
                self.file_path.name, self.window_size, missing,
            )
        return torch.tensor(weights, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        x = torch.tensor(self.windows[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx],  dtype=torch.long)
        return x, y

    @staticmethod
    def _load(path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(
                f"Split file not found: {path}. "
                "Run split_dataset.py first."
            )
        return pd.read_excel(path)

    @staticmethod
    def _rebuild_segments(df: pd.DataFrame) -> pd.DataFrame:
        """Re-derive segment boundaries from the saved split file.

        The saved files do not carry ``seg_id``.  A segment break is detected
        whenever ``Task``, ``Tractor Brand``, or the calendar date (extracted
        from ``Date and time``) changes between consecutive rows.

        Args:
            df: DataFrame loaded from a split Excel file.

        Returns:
            The same DataFrame with an added integer ``_seg_id`` column.
        """
        date = df["Date and time"].dt.normalize()  # midnight of each timestamp
        df["_seg_id"] = (
            (df[LABEL_COL]         != df[LABEL_COL].shift())         |
            (df["Tractor Brand"]   != df["Tractor Brand"].shift())    |
            (date                  != date.shift())
        ).cumsum()
        return df

    @staticmethod
    def _fit_or_reuse_encoder(
        df: pd.DataFrame,
        encoder: LabelEncoder | None,
    ) -> LabelEncoder:
        if encoder is not None:
            return encoder
        enc = LabelEncoder()
        enc.fit(df[LABEL_COL])
        return enc

    def _build_windows(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract all valid sliding windows from every segment.

        Iterates over segments in order.  For each segment of length ``L``,
        produces ``max(0, L - window_size + 1)`` windows.  Segments shorter
        than ``window_size`` are skipped without warning.

        Args:
            df: DataFrame with ``_seg_id``, feature columns, and ``LABEL_COL``.

        Returns:
            Tuple ``(windows, labels)`` where ``windows`` has shape
            ``(N, window_size, n_features)`` and ``labels`` has shape ``(N,)``.
        """
        windows_list: list[np.ndarray] = []
        labels_list:  list[int]        = []

        for _, seg in df.groupby("_seg_id", sort=True):
            features = seg[FEATURE_COLS].to_numpy(dtype=np.float32)
            # All rows in a segment share the same Task label.
            label = int(self.label_encoder.transform([seg[LABEL_COL].iloc[0]])[0])

            L = len(features)
            if L < self.window_size:
                # Segment too short to form even one window; skip.
                continue

            for start in range(L - self.window_size + 1):
                windows_list.append(features[start : start + self.window_size])
                labels_list.append(label)

        if not windows_list:
            raise ValueError(
                f"No windows could be extracted from {self.file_path.name} "
                f"with window_size={self.window_size}. "
                "All segments are shorter than window_size."
            )

        return (
            np.stack(windows_list, axis=0),
            np.array(labels_list, dtype=np.int64),
        )
