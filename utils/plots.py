import matplotlib.pyplot as plt
from pathlib import Path


def plot_train_metrics(history: dict, experiment_folder: Path) -> None:
    """
    This function plots the loss and error of the model.

    Params:
    -------

    history: dict
        history of the model contains the loss and error of the model values for each epoch
    experiment_folder: Path
        path to save the plot

    """

    plt.plot(history["loss"])
    plt.plot(history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.savefig(experiment_folder / "loss.png")
    plt.close()
    # plt.show()

    # plot model error
    plt.plot(history["mse"])
    plt.plot(history["val_mse"])
    plt.title("model error")
    plt.ylabel("error")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.savefig(experiment_folder / "error.png")
    plt.close()

def plot_xyz(X_train, X_test, experiment_folder: Path, filename: str="xyz_train_test") -> None:
    """
    This function plots the data distribuition.

    Params:
    --------
    X_train: np.ndarray
        X values for train
    X_test: np.ndarray
        X values for test
    experiment_folder: Path
        path to save the plot

    """

    x, y, z = X_train[:, -3], X_train[:, -2], X_train[:, -1]
    x2, y2, z2 = X_test[:, -3], X_test[:, -2], X_test[:, -1]

    plt.figure(figsize=(30, 8))
    plt.subplot(1, 3, 1)
    plt.scatter(x, y, label="xy_train")
    plt.scatter(x2, y2, label="xy_test")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.scatter(y, z, label="yz_train")
    plt.scatter(y2, z2, label="yz_test")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.scatter(x, z, label="xz_train")
    plt.scatter(x2, z2, label="xz_test")
    plt.legend()

    plt.savefig(experiment_folder / f"{filename}.png")
    plt.close()
