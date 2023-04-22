import numpy as np
import pandas as pd
from pprint import pprint

import tensorflow as tf
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Dropout  # type: ignore

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

np.set_printoptions(precision=6, suppress=True)


dh_table = np.array([
    [0.,   0.,          12.0,      0.],
    [0.,   0.,          18.0,      0.]
])


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


def fkine(theta, alpha, r, d, offset=np.array([0., 0.])):  # , 0, 0, 0, 0])):

    # print(f"Thetas = {theta}")
    # print(f"alphas = {alpha}")
    # print(f"rs = {r}")
    # print(f"ds = {d}\n")

    T = np.eye(4)
    theta = theta + offset

    for i in range(len(theta)):
        T_i = dh_transform(theta[i], alpha[i], r[i], d[i])
        T = np.dot(T, T_i)

    return T


def get_position(thetas):

    _dh_table = [thetas] + [dh_table[:, i] for i in range(1, 4)]
    matrix_t = fkine(*_dh_table)

    # pprint(matrix_t)

    xyz = matrix_t[:-1, -1]

    r20 = matrix_t[2, 0]
    r21 = matrix_t[2, 1]
    r22 = matrix_t[2, 2]
    r10 = matrix_t[1, 0]
    r00 = matrix_t[0, 0]

    # print(f"\nr20 = DH TABLE[2,0] = {r20}")
    # print(f"r21 = DH TABLE[2,1] = {r21}")
    # print(f"r22 = DH TABLE[2,2] = {r22}")
    # print(f"r10 = DH TABLE[1,0] = {r10}")
    # print(f"r00 = DH TABLE[0,0] = {r00}\n\n")

    pitch = np.arctan2(-r20, np.sqrt(r21**2 + r22**2))
    roll = np.arctan2(r10/np.cos(pitch), r00/np.cos(pitch))
    yaw = np.arctan2(r21/np.cos(pitch), r22/np.cos(pitch))

    # print("pitch = arcotan2(-r20, sqrt(r21^2 + r22^2))\n")

    # print("r21^2 = ", r21**2)
    # print("r22^2 = ", r22**2)
    # print("r21^2 + r22^2 = ", r21**2 + r22**2)
    # print("sqrt(r21^2 + r22^2) = ", np.sqrt(r21**2 + r22**2))
    # print("arcotan2(-r20, sqrt(r21^2 + r22^2)) = ", np.arctan2(-r20, np.sqrt(r21**2 + r22**2)))

    # print("\nroll = arcotan2(r10/cos(pitch), r00/cos(pitch))\n")
    # print(f"cos(pitch) = {np.cos(pitch)}")
    # print(f"r00/cos(pitch) = {r00/np.cos(pitch)}")
    # print(f"r10/cos(pitch) = {r10/np.cos(pitch)}")
    # print(f"arcotan2(r10/cos(pitch), r00/cos(pitch)) = {np.arctan2(r10/np.cos(pitch), r00/np.cos(pitch))}")

    # print("\nyaw = arcotan2(r21/cos(pitch), r22/cos(pitch))\n")
    # print(f"cos(pitch) = {np.cos(pitch)}")
    # print(f"r21/cos(pitch) = {r21/np.cos(pitch)}")
    # print(f"r22/cos(pitch) = {r22/np.cos(pitch)}")
    # print(f"arcotan2(r21/cos(pitch), r22/cos(pitch)) = {np.arctan2(r21/np.cos(pitch), r22/np.cos(pitch))}")

    # return [roll, pitch, yaw] + list(xyz)
    return xyz

# get_position(thetas[0, :])

# fkine(*[dh_table[:,i] for i in range(4)])

# for name, col in zip(["Thetas", "Alphas", "Ds", "As"], [dh_table[:,i] for i in range(4)]):
    # print(name, col)


get_position([np.pi/2])  # tratar esse erro!

"""### Puma560 angles rotations contraints"""

# INTERVALOS ORIGINAIS np.array([(-np.pi, np.pi), (-np.pi/2, np.pi/2) ])

# I1 [-180, 180]
# I2 [-90, 90]

# Particoes
# I1 [-180, 0] [0, 180] 2
# I2 [-90, 0]  [0, 90]  2

# Conjuntos 2x2 = 4
#

# PRIMEIRA PARTICAO

# np.array([[-2.792, 2.792], [-3.9269, 0.7854], [-0.7854, 3.9269], [-1.9198, 2.967], [-1.7453, 1.7453], [-4.6425, 4.6425]])
angle_ranges = np.array([(0, np.pi), (0, np.pi)])
angle_ranges

"""### Generate Dataset with N samples randomly"""

# n samples
N = 100

thetas1 = np.linspace(angle_ranges[0, 0], angle_ranges[0, 1], N)
thetas2 = np.linspace(angle_ranges[1, 0], angle_ranges[1, 1], N)

thetas = np.array([np.array([t1, t2]) for t1 in thetas1 for t2 in thetas2])

pprint(thetas)
pprint(thetas.shape)

positions = np.array(
    [get_position(theta) for theta in thetas]
)

len(positions)

pprint(positions.shape)

plt.figure(figsize=(16, 8))
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

"""# Model Design"""


def build_model(input_dim, output_dim):

    model = Sequential()
    model.add(Dense(units=10, activation='relu', input_dim=input_dim))
    model.add(Dense(units=20, activation='relu'))
    model.add(Dense(units=output_dim))

    return model


model_params = {
    'optimizer': 'adam',
    'loss': 'mse',
    'metrics': ['mse']
}

model = build_model(positions.shape[1], thetas.shape[1])
model.summary()

opt = tf.keras.optimizers.Adam(learning_rate=0.001)

# monitoring loss with ealty stop
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='loss', patience=10, min_delta=0.0001)

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

y_hat = model.predict(X_test[0].reshape(1, -1), verbose=0)[0]

print(f"Original  Thetas {y_test[0]}")
print(f"Predicted Thetas {y_hat}")

true_position = get_position(y_test[0])
pred_position = get_position(y_hat)

print(f"Position given by theta label {true_position}")
print(f"Position given by theta pred  {pred_position}")

"""#Testando algumas possibilidades"""
