import numpy as np

from robot import Robot3DOF
from matplotlib import pyplot as plt
from keras.models import load_model

if __name__ == "__main__":
    model = load_model(
        "runs/exp2/model.h5"
    )

    robot = Robot3DOF()

    p0 = np.array([0., 0., -0.1])
    dp = np.array([0., 0.,  0.1])

    # generate intermediate points between p0 and p1 = p0 + dp
    p1 = p0 + dp
    n_points = 50

    points = np.linspace(p0, p1, n_points)

    Q = model.predict(points)

    robot.show(Q, p=p1)

    # plot joints angles
    plt.figure()
    plt.plot(Q, linewidth=2)
    plt.legend(["q1", "q2", "q3"])
    plt.xlabel("time")
    plt.ylabel("joint angles")

    pos = np.array([robot.get_position(q) for q in Q])
    # plot end-effector position
    plt.figure()
    plt.plot(pos, linewidth=2)
    plt.plot(points, "--", linewidth=2)
    plt.legend(["x", "y", "z"])
    plt.xlabel("time")
    plt.ylabel("end-effector position")

    p1_pred = robot.get_position(Q[-1])

    print("p1: ", points[-1])
    print("p1_pred: ", p1_pred)

    plt.show()
