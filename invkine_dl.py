import numpy as np
import pandas as pd
from pprint import pprint

import tensorflow as tf
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense  # type: ignore

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


L1 = 1
L2 = 1


def dh_transform(theta, alpha, r, d):
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    T = np.array([[ct, -st*ca, st*sa, r*ct],
                  [st, ct*ca, -ct*sa, r*st],
                  [0, sa, ca, d],
                  [0, 0, 0, 1]])
    return T


def fkine(theta, alpha, r, d, offset=np.array([0., 0.])):
    T = np.eye(4)
    theta = theta + offset
    for i in range(len(theta)):
        T_i = dh_transform(theta[i], alpha[i], r[i], d[i])
        T = np.dot(T, T_i)
    return T


def get_position(thetas):
    # get_position([np.pi/2])  # tratar esse erro!
    _dh_table = [thetas] + [dh_table[:, i] for i in range(1, 4)]
    matrix_t = fkine(*_dh_table)
    xyz = matrix_t[:-1, -1]
    return xyz


def plot_arm(thetas):

    x1 = L1 * np.cos(thetas[0])
    y1 = L1 * np.sin(thetas[0])

    x2 = x1 + L2 * np.cos(thetas[0] + thetas[1])
    y2 = y1 + L2 * np.sin(thetas[0] + thetas[1])

    fig = plt.figure(figsize=(8, 8))

    plt.plot([0, x1], [0, y1], 'r', label='link 1', linewidth=5)
    plt.plot([x1, x2], [y1, y2], 'b', label='link 2', linewidth=5)
    plt.scatter([0, x1, x2], [0, y1, y2], c='k', s=50, zorder=10)
    plt.legend()
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.grid()

    return fig


if __name__ == "__main__":

    dh_table = np.array([
        [0.,   0.,          L1,      0.],
        [0.,   0.,          L2,      0.]
    ])

    angle_ranges = np.array([(0, np.pi), (0, np.pi)])

    N = 100  # N**2 points

    thetas1 = np.linspace(angle_ranges[0, 0], angle_ranges[0, 1], N)
    thetas2 = np.linspace(angle_ranges[1, 0], angle_ranges[1, 1], N)

    thetas = np.array([np.array([t1, t2]) for t1 in thetas1 for t2 in thetas2])

    positions = np.array(
        [get_position(theta) for theta in thetas]
    )

    plt.figure(figsize=(10, 8))
    plt.scatter(positions[:, 0], positions[:, 1], label='nuvem de pontos')
    plt.savefig('nuvem_de_pontos.png')
    plt.show()

    """### Data split"""

    X_train, X_test, y_train, y_test = train_test_split(
        positions, thetas, test_size=0.25, random_state=42)

    """# Data visualization"""

    x, y = X_train[:, 0], X_train[:, 1]
    x2, y2 = X_test[:, 0], X_test[:, 1]

    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.scatter(x, y, label='xy_train')
    plt.subplot(1, 2, 2)
    plt.scatter(x2, y2, label='xy_test')
    plt.legend()
    plt.savefig('xy_train_test.png')
    plt.show()

    input_dim, output_dim = positions.shape[1], thetas.shape[1]

    model = Sequential([
        Dense(units=32, activation='relu', input_dim=input_dim),
        Dense(units=64, activation='relu'),
        Dense(units=output_dim)
    ])

    model.summary()

    model_params = {
        'optimizer': 'adam',
        'loss': 'mse',
        'metrics': ['mse']
    }

    opt = tf.keras.optimizers.Adam(learning_rate=0.001)

    # monitoring loss with early stop
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='loss', patience=4, min_delta=0.0001)

    model.compile(
        optimizer=opt,
        loss=model_params['loss'],
        metrics=model_params['metrics']
    )

    history = model.fit(x=X_train, y=y_train, batch_size=128,
                        epochs=5000, validation_data=(X_test, y_test), callbacks=[early_stop])

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('loss.png')
    plt.show()

    # plot model error
    plt.plot(history.history['mse'])
    plt.plot(history.history['val_mse'])
    plt.title('model error')
    plt.ylabel('error')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('error.png')
    plt.show()

    model.evaluate(X_test, y_test)

    test_theta = np.array([np.pi/2, np.pi/2])
    test_pos = get_position(test_theta)

    y_hat = model.predict(test_pos.reshape(1, -1), verbose=0)[0]

    print(f"Original  Thetas {test_theta}")
    print(f"Predicted Thetas {y_hat}")

    pred_position = get_position(y_hat)

    print(f"Position given by theta label {test_pos}")
    print(f"Position given by theta pred  {pred_position}")

    fig1 = plot_arm(test_theta)
    plt.title('Desired arm position')
    fig2 = plot_arm(y_hat)
    plt.title('Predicted arm position')
    plt.show()
