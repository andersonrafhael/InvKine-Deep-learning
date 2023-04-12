import numpy as np
from pprint import pprint

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from robot import RobotPuma560
def plot_loss_val(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    return

def plot_xyz(X_train, X_test):
    x, y, z = X_train[:, -3], X_train[:, -2], X_train[:, -1]
    x2, y2, z2 = X_test[:, -3], X_test[:, -2], X_test[:, -1]

    plt.figure(figsize=(30, 8))
    plt.subplot(1,3,1)
    plt.scatter(x, y, label='xy_train')
    plt.scatter(x2, y2, label='xy_test')
    plt.legend()


    plt.subplot(1,3,3)
    plt.scatter(y, z, label='yz_train')
    plt.scatter(y2, z2, label='yz_test')
    plt.legend()

    plt.subplot(1,3,2)
    plt.scatter(x, z, label='xz_train')
    plt.scatter(x2, z2, label='xz_test')
    plt.legend()

    plt.show()
    return

def build_model(input_dim, output_dim):

    model = Sequential()
    model.add(Dense(units=32, activation='relu', input_dim=input_dim))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(units=output_dim))

    return model

np.set_printoptions(precision=6, suppress=True)

dh_table = np.array([
    
    [np.pi/2,   -np.pi/2,          0,      0.67183],
    [      0,           0,    0.43180,     0.13970],
    [      0,     np.pi/2,   -0.02032,           0],
    [      0,    -np.pi/2,         0,      0.43180],
    [      0,     np.pi/2,         0,            0],
    [      0,           0,         0,      0.05650]
    
])

robot = RobotPuma560(dh_table)

# Puma560 angles rotations contraints
angle_ranges = np.array([(-np.pi, np.pi)])

#n samples
N = 10000

# thethas = [theta_joint0, theta_joint1, theta_joint2, theta_joint3, theta_joint4, theta_joint5]
thetas = np.array([
    np.random.uniform(low=low, high=high, size=N) for i, (low, high) in enumerate(angle_ranges)
]).T

positions = np.array(
    [robot.get_position(theta) for theta in thetas]
)

X_train, X_test, y_train, y_test = train_test_split(positions, thetas, test_size=0.10, random_state=42)

plot_xyz(X_train, X_test)

model_params = {
    'optimizer':'adam',
    'loss':'mse',
    'metrics': ['mse']
}

model = build_model(positions.shape[1], thetas.shape[1])
model.summary()

opt = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(
    optimizer=opt, 
    loss=model_params['loss'],
    metrics=model_params['metrics']
)

history = model.fit(x=X_train, y=y_train, batch_size=128, epochs=50, validation_data=(X_test,y_test))
plot_loss_val(history)

model.evaluate(X_test, y_test)

y_hat = model.predict(X_test[0].reshape(1,-1), verbose=0)[0]

print(f"Original  Thetas {y_test[0]}")
print(f"Predicted Thetas {y_hat}")

true_position = robot.get_position(y_test[0])
pred_position = robot.get_position(y_hat)

print(f"Position given by theta label {true_position}")
print(f"Position given by theta pred  {pred_position}")