import matplotlib.pyplot as plt

from robot import Robot3DOF
from keras.models import load_model

if __name__ == "__main__":
    model = load_model(
        "runs/exp6/model.h5"
    )
    q_test = [0.3, 0.3, 0.3]
    robot = Robot3DOF()
    xyz_test = robot.get_position(q_test)
    pred_q = model.predict(xyz_test.reshape(1, 3))
    print(q_test, pred_q)
    robot.show(q=q_test)
    robot.show(q=pred_q[0])
