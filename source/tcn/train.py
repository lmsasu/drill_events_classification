import os
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.metrics import AUC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tcn import TCN
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

dict_hyperparams = {
    "lookback_window": 25,
    "batch_size": 32,
    "epochs": 1,
    "nb_filters": 64,
    "nb_stacks": 10,
    "kernel_size": 3,
    "use_batch_norm": True,
    "activation": "relu",
    "optimizer": "adamw",
}


def plot_save(history, directory_name, epochs, save_eps=False) -> None:
    os.makedirs(directory_name, exist_ok=True)
    print("Plotting training history")
    score_names = list(history.history.keys())
    count_perfomance_scores = len(score_names) // 2
    epoch_range = range(1, epochs + 1)
    for i in range(count_perfomance_scores):
        score_name = score_names[i]
        if score_name == "f1_score":
            continue
        plt.figure(figsize=(12, 6))
        plt.xticks(epoch_range)
        plt.plot(epoch_range, history.history[score_names[i]], label=score_names[i])
        plt.plot(
            epoch_range,
            history.history[score_names[i + count_perfomance_scores]],
            label=score_names[i + count_perfomance_scores],
        )
        plt.title(f"Training vs validation {score_name}")
        plt.ylabel(score_name)
        plt.xlabel("Epoch")
        plt.legend()
        plt.savefig(
            f"{os.path.join(directory_name, score_name)}.png", bbox_inches="tight"
        )
        if save_eps:
            plt.savefig(
                f"{os.path.join(directory_name, score_name)}.eps", bbox_inches="tight"
            )
        plt.show()


def write_predictions(model, x, y_groundtruth, original_labels, dir_name, subset_type):
    full_name = os.path.join(dir_name, f"{subset_type}_predictions.xlsx")
    predicted = model.predict(x)
    df_out = pd.DataFrame(x.reshape(x.shape[0], -1))
    df_out["predicted"] = np.argmax(predicted, axis=1)
    df_out["y_groundtruth"] = np.argmax(y_groundtruth, axis=1)
    df_out.to_excel(full_name, index=False)


def play(
    df: pd.DataFrame,
    verbose: bool = False,
    save_eps: bool = False,
    **hyperparameters,
):
    print("Play function called with hyperparameters:", hyperparameters)

    dir_root_results = "results"
    timestamp = str(pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S"))
    dir_name = os.path.join(dir_root_results, f"results_{timestamp}")
    os.makedirs(dir_name, exist_ok=True)

    with open(os.path.join(dir_name, "hyperparameters.txt"), "wt") as f:
        f.write("Hyperparameters:\n")
        for key, value in hyperparameters.items():
            f.write(f"\t{key}: {value}\n")

    output_column = "Task"
    df[output_column] = df[output_column].fillna("OPR")
    features = df.drop(columns=["Task"]).values

    if verbose:
        unique_text = df[output_column].dropna().unique()
        print(f"Unique text strings in {output_column}:", unique_text)
        print(df.head())
        print("Column names:", df.columns)
        nan_rows = df[df[output_column].isna()]
        print(f"Rows with NaN values in column {output_column}:")
        print("rows with nans:\n", nan_rows)

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(df["Task"])
    labels = to_categorical(encoded_labels)

    if verbose:
        print("Feature shape:", features.shape)
        print("Label shape:", labels.shape)
        print("Labels:", encoded_labels)

    lookback_window = hyperparameters["lookback_window"]
    x, y = [], []
    for i in range(lookback_window, len(labels)):
        x.append(features[i - lookback_window : i])
        y.append(labels[i])
    x, y = np.array(x), np.array(y)

    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=0.3, random_state=42
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.5, random_state=42
    )

    if verbose:
        print(f"{x_train.shape=}, {y_train.shape=}")
        print(f"{x_val.shape=}, {y_val.shape=}")
        print(f"{x_test.shape=}, {y_test.shape=}")

    # TensorBoard scalar callback fails with multi-label metrics (shape != scalar).
    # Use a per-epoch CSV logger instead to avoid the ValueError from tensorboard/plugins/scalar/summary_v2.py.
    csv_logger = keras.callbacks.CSVLogger(os.path.join(dir_name, "training_log.csv"))

    i = Input(shape=(lookback_window, features.shape[1]))
    m = TCN(
        use_batch_norm=hyperparameters["use_batch_norm"],
        nb_stacks=hyperparameters["nb_stacks"],
        nb_filters=hyperparameters["nb_filters"],
        kernel_size=hyperparameters["kernel_size"],
        activation=hyperparameters["activation"],
    )(i)
    m = Dense(labels.shape[1], activation="softmax")(m)
    model = Model(inputs=i, outputs=m)

    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        AUC(multi_label=True, num_labels=labels.shape[1]),
    ]
    model.compile(
        optimizer=hyperparameters["optimizer"],
        loss="categorical_crossentropy",
        metrics=metrics,
        jit_compile=False,
    )

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=hyperparameters["epochs"],
        batch_size=hyperparameters["batch_size"],
        callbacks=[csv_logger],
    )

    plot_save(history, dir_name, hyperparameters["epochs"], save_eps=save_eps)

    y_pred_test = model.predict(x_test)
    y_pred_classes_test = np.argmax(y_pred_test, axis=1)
    y_true_classes_test = np.argmax(y_test, axis=1)

    original_labels = label_encoder.inverse_transform(
        np.arange(len(label_encoder.classes_))
    )
    print("Original Labels:", original_labels)

    print(
        "Classification Report:\n",
        classification_report(y_true_classes_test, y_pred_classes_test),
    )
    with open(os.path.join(dir_name, "classification_report.txt"), "wt") as f:
        f.write("Classification Report:")
        f.write(classification_report(y_true_classes_test, y_pred_classes_test))

    cm = confusion_matrix(y_true_classes_test, y_pred_classes_test)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        cmap="Reds",
        fmt="d",
        xticklabels=original_labels,
        yticklabels=original_labels,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix on Test Data")
    _confusion_matrix_filename = "confusion_matrix"
    plt.savefig(
        f"{os.path.join(dir_name, _confusion_matrix_filename)}.png", bbox_inches="tight"
    )
    if save_eps:
        plt.savefig(
            f"{os.path.join(dir_name, _confusion_matrix_filename)}.eps",
            bbox_inches="tight",
        )
    plt.show()

    all_results_path = os.path.join(dir_root_results, "all_results.xlsx")
    if os.path.exists(all_results_path):
        df_results = pd.read_excel(all_results_path)
    else:
        lst_header = (
            ["timestamp"] + list(hyperparameters.keys()) + list(history.history.keys())
        )
        df_results = pd.DataFrame(columns=lst_header)

    lst_row = [timestamp]
    for key in hyperparameters.keys():
        lst_row.append(str(hyperparameters[key]))
    for metric in history.history.keys():
        lst_row.append(str(history.history[metric][-1]))
    df_results.loc[len(df_results)] = lst_row
    df_results.to_excel(all_results_path, index=False)

    print("Writing train predictions...")
    write_predictions(model, x_train, y_train, original_labels, dir_name, "train")
    print("Writing val predictions...")
    write_predictions(model, x_val, y_val, original_labels, dir_name, "val")
    print("Writing test predictions...")
    write_predictions(model, x_test, y_test, original_labels, dir_name, "test")

    return history, x_test, y_test, y_pred_test, y_pred_classes_test, model


if __name__ == "__main__":
    print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))

    dataset_path = Path(__file__).resolve().parents[2] / "data" / "DataBaseTCN.xlsx"
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. "
            f"Current working directory is {Path.cwd()}."
        )

    df = pd.read_excel(dataset_path)
    history, x_test, y_test, y_pred_test, y_pred_classes_test, model = play(
        df, **dict_hyperparams
    )
