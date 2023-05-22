import json
import argparse
import numpy as np
import tensorflow as tf

from keras.layers import Dense
from keras.models import Sequential, save_model

from sklearn.model_selection import train_test_split
from robot import Robot3DOF

from pathlib import Path
from utils.plots import plot_train_metrics, plot_xyz

from utils.experiments import (
    get_experiment_id,
    get_network_config,
    get_experiment_config,
    export_H_model
)


def build_model(
    net_config: dict, experiment_config: dict, experiment_folder: str
) -> tf.keras.Sequential:

    """
    This function builds the model based on the network configuration
    and saves the experiment configuration.

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
        "0": {
            "units": net_config[1],
            "activation": "relu",
            "input_shape": net_config[0],
        }
    }

    for i, units in enumerate(net_config[2:]):
        activ = "linear" if (i + 2) == last_idx_Layer else "relu"
        model.add(Dense(units, activation=activ))
        experiment_config["network_config"].update(
            {str(i + 2): {"units": units, "activation": activ}}
        )

    model.summary()

    with open(experiment_folder / "network_config", "w") as f:
        json.dump(experiment_config, f, indent=4)

    return model


np.set_printoptions(precision=6, suppress=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hidden-layers-config",
        type=int,
        nargs="+",
        default=[16, 16],
        help="hidden layers config to build neural network",
    )
    parser.add_argument(
        "--epochs", type=int,
        default=500, help="number of epochs to train the model"
    )
    parser.add_argument(
        "--bsize", type=int,
        default=128, help="batch size to train the model"
    )
    parser.add_argument(
        "--lrate", type=float,
        default=0.005, help="learning rate to train the model"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=200,
        help="number of samples to generate the dataset",
    )
    args = parser.parse_args()

    ROOT_DIR = Path(__file__).absolute().parent

    experiment_id = get_experiment_id(ROOT_DIR)
    experiment_folder = ROOT_DIR / Path(f"runs/exp{experiment_id}")
    experiment_folder.mkdir(parents=True, exist_ok=True)

    experiment_config = get_experiment_config(args)

    robot = Robot3DOF()

    angle_ranges = np.array([(0, np.pi), (0, np.pi)])

    n = args.n_samples
    theta1 = np.linspace(*angle_ranges[0], n)
    theta2 = np.linspace(*angle_ranges[1], n)
    # theta3 = np.linspace(*angle_ranges[2], n)

    # Divides each interval in n parts and then generates n**2 samples
    # thetas = np.array(np.meshgrid(theta1, theta2, theta3)).T.reshape(-1, 3)
    thetas = np.array(np.meshgrid(theta1, theta2)).T.reshape(-1, 2)

    # Generate, split and plot dataset
    positions = np.array([robot.get_position(theta) for theta in thetas])

    X_train, X_test, y_train, y_test = train_test_split(
        positions, thetas, test_size=0.25, random_state=42
    )
    plot_xyz(X_train, X_test, experiment_folder)

    # Build and train model

    model_config = get_network_config(
        positions.shape[1], thetas.shape[1], args.hidden_layers_config
    )
    model_params = {
        "optimizer": tf.keras.optimizers.Adam(learning_rate=args.lrate),
        "loss": "mse",
        "metrics": ["mse", "mae"],
    }

    model = build_model(model_config, experiment_config, experiment_folder)
    model.summary()

    model.compile(
        optimizer=model_params["optimizer"],
        loss=model_params["loss"],
        metrics=model_params["metrics"],
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="loss", patience=3, min_delta=0.01
    )

    history = model.fit(
        x=X_train,
        y=y_train,
        batch_size=args.bsize,
        epochs=args.epochs,
        validation_data=(X_test, y_test),
    )

    # Plot model train metrics
    plot_train_metrics(history.history, experiment_folder)

    # Evaluate model

    model.evaluate(X_test, y_test)
    y_hat = model.predict(X_test[0].reshape(1, -1), verbose=0)[0]

    print(f"Original  Thetas {y_test[0]}")
    print(f"Predicted Thetas {y_hat}")

    true_position = robot.get_position(y_test[0])
    pred_position = robot.get_position(y_hat)

    print(f"Position given by theta label {true_position}")
    print(f"Position given by theta pred  {pred_position}")

    export_H_model(model, experiment_folder)
    save_model(model, experiment_folder / "model.h5")
