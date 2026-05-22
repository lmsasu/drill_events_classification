import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
logger = logging.getLogger(__name__)


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            f"Current working directory is {Path.cwd()}."
        )
    return pd.read_excel(path)


def prepare_data(
    df: pd.DataFrame,
    lookback_window: int,
    verbose: bool = False,
):
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

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(df[output_column])
    labels = np.eye(encoded_labels.max() + 1, dtype=np.float32)[encoded_labels]

    if verbose:
        logger.info("Feature shape: %s", features.shape)
        logger.info("Label shape: %s", labels.shape)
        logger.info("Encoded labels: %s", encoded_labels)

    x, y = [], []
    for i in range(lookback_window, len(labels)):
        x.append(features[i - lookback_window : i])
        y.append(labels[i])
    x, y = np.array(x), np.array(y)

    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=0.3, random_state=42, stratify=y
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    if verbose:
        logger.info("x_train.shape=%s, y_train.shape=%s", x_train.shape, y_train.shape)
        logger.info("x_val.shape=%s, y_val.shape=%s", x_val.shape, y_val.shape)
        logger.info("x_test.shape=%s, y_test.shape=%s", x_test.shape, y_test.shape)

    return x_train, x_val, x_test, y_train, y_val, y_test, label_encoder
