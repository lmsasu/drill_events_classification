import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
logger = logging.getLogger(__name__)


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the dataset from an Excel file.

    Args:
        path: Absolute or relative path to the ``.xlsx`` file.

    Returns:
        Raw DataFrame with all columns intact.

    Raises:
        FileNotFoundError: If no file exists at ``path``.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            f"Current working directory is {Path.cwd()}."
        )
    return pd.read_excel(path)


def prepare_data(
    df: pd.DataFrame,
    lookback_window: int,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    verbose: bool = False,
):
    """Build sliding-window arrays and split them into train, val, and test sets.

    The raw time series is partitioned **chronologically** before any windows
    are constructed, eliminating both overlap leakage (adjacent windows sharing
    raw time steps across splits) and temporal leakage (future observations
    appearing in the training set).  Validation and test windows are allowed to
    look back into the tail of the preceding segment for context, but their
    labelled endpoints are strictly confined to their own period.

    The ``LabelEncoder`` is fitted on the full raw series so that every class
    is known prior to splitting; only the encoded arrays are partitioned.

    Args:
        df: Raw dataset DataFrame.  Must contain a ``"Task"`` column with
            string class labels; all other columns are treated as features.
            Missing values in ``"Task"`` are filled with ``"OPR"``.
        lookback_window: Number of consecutive time steps that form one input
            window (sequence length fed to the model).
        val_ratio: Fraction of raw time steps reserved for validation.
        test_ratio: Fraction of raw time steps reserved for testing.
        verbose: When ``True``, log dataset statistics and split sizes.

    Returns:
        A tuple ``(x_train, x_val, x_test, y_train, y_val, y_test,
        label_encoder)`` where:

        - ``x_*`` are float32 arrays of shape ``(n_samples, lookback_window,
          n_features)``.
        - ``y_*`` are float32 one-hot arrays of shape ``(n_samples,
          n_classes)``.
        - ``label_encoder`` is the fitted :class:`sklearn.preprocessing.LabelEncoder`
          mapping integer indices back to original class names.
    """
    output_column = "Task"
    df = df.copy()
    df[output_column] = df[output_column].fillna("OPR")
    features = df.drop(columns=[output_column]).values

    if verbose:
        unique_text = df[output_column].dropna().unique()
        logger.info("Unique text strings in %s: %s", output_column, unique_text)
        logger.info("Head:\n%s", df.head())
        logger.info("Column names: %s", list(df.columns))
        nan_rows = df[df[output_column].isna()]
        logger.info("Rows with NaN values in column %s:\n%s", output_column, nan_rows)

    # LabelEncoder is fit on all raw rows so every class is known before splitting.
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(df[output_column])
    labels = np.eye(encoded_labels.max() + 1, dtype=np.float32)[encoded_labels]

    if verbose:
        logger.info("Feature shape: %s", features.shape)
        logger.info("Label shape: %s", labels.shape)

    # Temporal split: cut the raw series chronologically so that no window
    # endpoint from one split ever appears in another split.
    # Val/test windows may look back into the tail of the previous segment
    # (historical context) but their labeled endpoints are strictly within
    # their own period, preventing any future-data leakage.
    n = len(labels)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_val - n_test

    def _make_windows(feat: np.ndarray, lab: np.ndarray):
        """Slide a window of length ``lookback_window`` over ``feat``/``lab``.

        Args:
            feat: Feature array of shape ``(T, n_features)``.
            lab: One-hot label array of shape ``(T, n_classes)``.

        Returns:
            Tuple ``(x, y)`` where ``x`` has shape
            ``(T - lookback_window, lookback_window, n_features)`` and ``y``
            has shape ``(T - lookback_window, n_classes)``.
        """
        xs, ys = [], []
        for i in range(lookback_window, len(lab)):
            xs.append(feat[i - lookback_window : i])
            ys.append(lab[i])
        return np.array(xs), np.array(ys)

    x_train, y_train = _make_windows(features[:n_train], labels[:n_train])

    # Include up to lookback_window rows of train tail as context for val.
    val_ctx = max(0, n_train - lookback_window)
    x_val, y_val = _make_windows(
        features[val_ctx : n_train + n_val],
        labels[val_ctx : n_train + n_val],
    )

    # Include up to lookback_window rows of val tail as context for test.
    test_ctx = max(0, n_train + n_val - lookback_window)
    x_test, y_test = _make_windows(features[test_ctx:], labels[test_ctx:])

    if verbose:
        logger.info("Raw split sizes — train: %d, val: %d, test: %d", n_train, n_val, n_test)
        logger.info("x_train.shape=%s, y_train.shape=%s", x_train.shape, y_train.shape)
        logger.info("x_val.shape=%s, y_val.shape=%s", x_val.shape, y_val.shape)
        logger.info("x_test.shape=%s, y_test.shape=%s", x_test.shape, y_test.shape)

    return x_train, x_val, x_test, y_train, y_val, y_test, label_encoder
