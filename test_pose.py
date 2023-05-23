import numpy as np

from robot import Robot3DOF
from keras.models import load_model

if __name__ == "__main__":
    model = load_model(
        "runs/exp2/model.h5"
    )

    robot = Robot3DOF()

    angles = np.array([1.57, 1.57, 1.57])
    position = robot.get_position(angles)

    pred_angles = model.predict(position.reshape(1, 3))[0]
    pred_position = robot.get_position(pred_angles)

    print("angles: ", angles)
    print("position: ", position)
    print("pred_angles: ", pred_angles)
    print("pred_position: ", pred_position)

    robot.robot.plot(
        angles, limits=[-0.3, 0.3, -0.3, 0.3, -0.3, 0.3]
    )
    robot.robot.plot(
        pred_angles, limits=[-0.3, 0.3, -0.3, 0.3, -0.3, 0.3], loop=True)
