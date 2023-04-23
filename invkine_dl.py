import numpy as np
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense  # type: ignore

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import argparse


import json

L1 = 1
L2 = 1


def dh_transform(theta: float, alpha: float, r: float, d: float) -> np.ndarray:
    """
    This function returns the homogeneous transformation matrix for a given set of DH parameters.

    Params:
        theta: float
            rotation about z-axis
        alpha: float
            rotation about x-axis
        r: float
            translation along x-axis
        d: float
            translation along z-axis

    Returns:
        T: np.ndarray
            homogeneous transformation matrix

    """
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    T = np.array(
        [
            [ct, -st * ca, st * sa, r * ct],
            [st, ct * ca, -ct * sa, r * st],
            [0, sa, ca, d],
            [0, 0, 0, 1],
        ]
    )
    return T


def fkine(
    theta: np.ndarray,
    alpha: np.ndarray,
    r: np.ndarray,
    d: np.ndarray,
    offset=np.array([0.0, 0.0]),
) -> np.ndarray:
    """
    This function returns the homogeneous transformation matrix for a given set of DH parameters of T0 to Tn.

    Params:
        theta: np.ndarray
            rotation about z-axis
        alpha: np.ndarray
            rotation about x-axis
        r: np.ndarray
            translation along x-axis
        d: np.ndarray
            translation along z-axis
        offset: np.ndarray
            offset to be added to theta

    Returns:
        T: np.ndarray
            homogeneous transformation matrix for T0 to Tn

    """
    T = np.eye(4)
    theta = theta + offset

    for i in range(len(theta)):
        T_i = dh_transform(theta[i], alpha[i], r[i], d[i])
        T = np.dot(T, T_i)
    return T


def get_position(thetas: np.ndarray) -> np.ndarray:
    """
    This function returns the position of the end effector for a given set of joint angles.

    Params:
        thetas: np.ndarray
            joint angles

    Returns:
        xyz: np.ndarray
            position of the end effector: x, y, z
    """
    # get_position([np.pi/2])  # tratar esse erro!

    _dh_table = [thetas] + [dh_table[:, i] for i in range(1, 4)]
    matrix_t = fkine(*_dh_table)
    xyz = matrix_t[:-1, -1]
    return xyz


def plot_arm(thetas: np.ndarray) -> plt.figure:
    """
    This function plots the arm for a given set of joint angles.

    Params:
        thetas: np.ndarray
            joint angles

    Return:
        fig: matplotlib.figure.Figure
            figure of plotted arm
    """
    x1 = L1 * np.cos(thetas[0])
    y1 = L1 * np.sin(thetas[0])

    x2 = x1 + L2 * np.cos(thetas[0] + thetas[1])
    y2 = y1 + L2 * np.sin(thetas[0] + thetas[1])

    fig = plt.figure(figsize=(8, 8))

    plt.plot([0, x1], [0, y1], "r", label="link 1", linewidth=5)
    plt.plot([x1, x2], [y1, y2], "b", label="link 2", linewidth=5)
    plt.scatter([0, x1, x2], [0, y1, y2], c="k", s=50, zorder=10)
    plt.legend()
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.grid()

    return fig


def plot_train_metrics(history: tf.keras.callbacks.History) -> None:
    """
    This function plots the loss and error of the model.

    Params:
        history: tf.keras.callbacks.History
            history of the model contains the loss and error of the model values for each epoch

    """

    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.savefig(Path("plots") / "loss.png")
    plt.close()
    # plt.show()

    # plot model error
    plt.plot(history.history["mse"])
    plt.plot(history.history["val_mse"])
    plt.title("model error")
    plt.ylabel("error")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.savefig(Path("plots") / "error.png")
    plt.close()


def get_experiment_id() -> Path:
    """
    This function returns the path of the next experiment to be saved.

    Params:
        None

    Returns:
        ith_experiment_path: Path
            path of the next experiment to be saved

    """
    i = 0
    while True:
        ith_experiment_path = Path("experiments") / f"experiment_{i}.json"
        if not ith_experiment_path.exists():
            break
        i += 1

    return Path("experiments") / f"experiment_{i}.json"


def build_model(net_config: dict, experiment_config: dict) -> tf.keras.Sequential:
    """
    This function builds the model based on the network configuration and saves the experiment configuration.

    Params:

        net_config: dict
            network configuration with the number of units for each layer
        experiment_config: dict
            experiment configuration with the number of epochs, batch size

    Returns:
        model: tf.keras.Sequential
            model built based on the network configuration
    """

    last_idx_Layer = len(net_config) - 1

    model = Sequential()
    model.add(Dense(net_config[1], activation="relu", input_dim=net_config[0]))
    experiment_config["network_config"] = {
        "0": {"units": net_config[1], "activation": "relu", "i_shape": net_config[0]}
    }

    for i, units in enumerate(net_config[2:]):
        activ = "linear" if (i + 2) == last_idx_Layer else "relu"
        model.add(Dense(units, activation=activ))
        experiment_config["network_config"].update(
            {str(i + 2): {"units": units, "activation": activ}}
        )

    model.summary()

    with open(get_experiment_id(), "w") as f:
        json.dump(experiment_config, f, indent=4)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hidden-layers-config",
        type=int,
        nargs="+",
        default=[10, 20],
        help="hidden layers config to build neural network",
    )
    parser.add_argument(
        "--epochs", type=int, default=5000, help="number of epochs to train the model"
    )
    parser.add_argument(
        "--bsize", type=int, default=128, help="batch size to train the model"
    )
    parser.add_argument(
        "--lrate", type=float, default=0.001, help="learning rate to train the model"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="number of samples to generate the dataset",
    )
    args = parser.parse_args()

    Path("plots").mkdir(parents=True, exist_ok=True)
    Path("experiments").mkdir(parents=True, exist_ok=True)

    experiment_config = {
        "training_config": {
            "epochs": args.epochs,
            "batch_size": args.bsize,
            "learning_rate": args.lrate,
            "n_samples": args.n_samples,
        },
    }

    dh_table = np.array([[0.0, 0.0, L1, 0.0], [0.0, 0.0, L2, 0.0]])
    angle_ranges = np.array([(0, np.pi), (0, np.pi)])
    N = args.n_samples

    thetas1 = np.linspace(angle_ranges[0, 0], angle_ranges[0, 1], N)
    thetas2 = np.linspace(angle_ranges[1, 0], angle_ranges[1, 1], N)

    thetas = np.stack(np.meshgrid(thetas1, thetas2), axis=-1).reshape(-1, 2)
    positions = np.array([get_position(theta) for theta in thetas])

    plt.figure(figsize=(10, 8))
    plt.scatter(positions[:, 0], positions[:, 1], label="nuvem de pontos")
    plt.savefig(Path("plots") / "nuvem_de_pontos.png")
    plt.close()

    """### Data split"""

    X_train, X_test, y_train, y_test = train_test_split(
        positions, thetas, test_size=0.25, random_state=42
    )

    """# Data visualization"""

    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.scatter(X_train[:, 0], X_train[:, 1], label="xy_train")
    plt.subplot(1, 2, 2)
    plt.scatter(X_test[:, 0], X_test[:, 1], label="xy_test")
    plt.legend()
    plt.savefig(Path("plots") / "xy_train_test.png")
    plt.close()

    input_dim, output_dim = positions.shape[1], thetas.shape[1]
    # [input_shape, units_input_layer, units_2th_layer, ..., units_output_layer]
    net_config = [input_dim] + args.hidden_layers_config + [output_dim]

    model = build_model(net_config, experiment_config)
    model_params = {"optimizer": "adam", "loss": "mse", "metrics": ["mse"]}
    opt = tf.keras.optimizers.Adam(learning_rate=args.lrate)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="loss", patience=4, min_delta=0.0001
    )

    model.compile(
        optimizer=opt, loss=model_params["loss"], metrics=model_params["metrics"]
    )

    history = model.fit(
        x=X_train,
        y=y_train,
        batch_size=args.bsize,
        epochs=args.epochs,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
    )

    plot_train_metrics(history)

    model.evaluate(X_test, y_test)

    test_theta = np.array([np.pi / 2, np.pi / 2])
    test_pos = get_position(test_theta)

    y_hat = model.predict(test_pos.reshape(1, -1), verbose=0)[0]

    print(f"Original  Thetas {test_theta}")
    print(f"Predicted Thetas {y_hat}")

    pred_position = get_position(y_hat)

    print(f"Position given by theta label {test_pos}")
    print(f"Position given by theta pred  {pred_position}")

    fig1 = plot_arm(test_theta)
    plt.title("Desired arm position")
    fig2 = plot_arm(y_hat)
    plt.title("Predicted arm position")
    # plt.show()
