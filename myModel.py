from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.model_selection import train_test_split
from utils.plots import plot_train_metrics

from robot import RobotPuma560

if __name__ == "__main__":
    
    dh_table = np.array([[0.0, 0.0, 1, 0.0], [0.0, 0.0, 1, 0.0]])

    robot = RobotPuma560(dh_table)

    # Puma560 angles rotations contraints
    # angle_ranges = np.array([(-np.pi, np.pi)])
    angle_ranges = np.array([(0, np.pi), (0, np.pi)])

    # thethas = [theta_joint0, theta_joint1, theta_joint2, theta_joint3, theta_joint4, theta_joint5]

    thetas = np.array(
        [
            np.random.uniform(low=low, high=high, size=100)
            for _, (low, high) in enumerate(angle_ranges)
        ]
    ).T

    ### Generate, split and plot dataset

    positions = np.array([robot.get_position(theta) for theta in thetas])

    X_train, X_test, y_train, y_test = train_test_split(
        positions, thetas, test_size=0.2, random_state=42
    )
    
    
    input_layer = keras.Input(shape=(positions.shape[1],))
    
    x = layers.Dense(10, activation="relu")(input_layer)
    x = layers.Dense(30, activation="relu")(x)
    x = layers.Dense(50, activation="relu")(x)
    
    y0 = layers.Dense(1, name="theta0")(x)
    y1 = layers.Dense(1, name="theta1")(x)
    
    combined_output_layer = layers.concatenate([y0, y1])
    reshaped_output_layer = layers.Reshape((2,))(combined_output_layer)
    model = keras.Model(inputs=input_layer, outputs=reshaped_output_layer)
    model.summary()
    
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="loss", patience=10, min_delta=0.0001
    )
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.005), loss="mse", metrics=["mse", "mae"])
    history = model.fit(X_train, y_train, epochs=500, batch_size=128, validation_data=(X_test, y_test), callbacks=[early_stop])
    
    Path("output").mkdir(parents=True, exist_ok=True)
    
    plot_train_metrics(history.history, Path("output"))
    
    model.save("output/model.h5")