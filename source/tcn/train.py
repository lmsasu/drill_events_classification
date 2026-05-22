import logging
import os
import sys
from pathlib import Path

# Must be set before TensorFlow is imported.
# Disables the cuDNN v8/v9 frontend graph API, which fails on Pascal (SM 6.1)
# with cuDNN 9.x ("No algorithm worked" / CUDNN_STATUS_EXECUTION_FAILED).
os.environ.setdefault("TF_ENABLE_CUDNN_FRONTEND", "0")
# Prevent TF from pre-allocating the entire GPU VRAM at startup.
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.metrics import AUC
from sklearn.metrics import classification_report, confusion_matrix
from tcn import TCN
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared_modules.data_preparation import load_dataset, prepare_data
from shared_modules.logging_setup import setup_logging

logger = logging.getLogger(__name__)


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
    logger.info("Plotting training history")
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
    verbose: bool = True,
    save_eps: bool = False,
    **hyperparameters,
):
    logger.info("Play function called with hyperparameters: %s", hyperparameters)

    dir_root_results = "results"
    timestamp = str(pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S"))
    dir_name = os.path.join(dir_root_results, f"results_{timestamp}")
    os.makedirs(dir_name, exist_ok=True)

    with open(os.path.join(dir_name, "hyperparameters.txt"), "wt") as f:
        f.write("Hyperparameters:\n")
        for key, value in hyperparameters.items():
            f.write(f"\t{key}: {value}\n")

    lookback_window = hyperparameters["lookback_window"]
    x_train, x_val, x_test, y_train, y_val, y_test, label_encoder = prepare_data(
        df, lookback_window, verbose=verbose
    )

    # TensorBoard scalar callback fails with multi-label metrics (shape != scalar).
    # Use a per-epoch CSV logger instead to avoid the ValueError from tensorboard/plugins/scalar/summary_v2.py.
    csv_logger = keras.callbacks.CSVLogger(os.path.join(dir_name, "training_log.csv"))

    i = Input(shape=(lookback_window, x_train.shape[2]))
    m = TCN(
        use_batch_norm=hyperparameters["use_batch_norm"],
        nb_stacks=hyperparameters["nb_stacks"],
        nb_filters=hyperparameters["nb_filters"],
        kernel_size=hyperparameters["kernel_size"],
        activation=hyperparameters["activation"],
    )(i)
    m = Dense(y_train.shape[1], activation="softmax")(m)
    model = Model(inputs=i, outputs=m)

    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        AUC(multi_label=True, num_labels=y_train.shape[1]),
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
    logger.info("Original Labels: %s", original_labels)

    logger.info(
        "Classification Report:\n%s",
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

    logger.info("Writing train predictions...")
    write_predictions(model, x_train, y_train, original_labels, dir_name, "train")
    logger.info("Writing val predictions...")
    write_predictions(model, x_val, y_val, original_labels, dir_name, "val")
    logger.info("Writing test predictions...")
    write_predictions(model, x_test, y_test, original_labels, dir_name, "test")

    return history, x_test, y_test, y_pred_test, y_pred_classes_test, model


if __name__ == "__main__":
    setup_logging()
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logger.info("Num GPUs Available: %d", len(gpus))

    dataset_path = Path(__file__).resolve().parents[2] / "data" / "DataBaseTCN.xlsx"
    df = load_dataset(dataset_path)
    history, x_test, y_test, y_pred_test, y_pred_classes_test, model = play(
        df, verbose=True, **dict_hyperparams
    )
