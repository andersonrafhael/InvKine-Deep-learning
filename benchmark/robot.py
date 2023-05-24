import numpy as np

from roboticstoolbox import DHRobot, RevoluteDH


class Robot3DOF:
    def __init__(self) -> None:
        self.robot = DHRobot(
            [
                RevoluteDH(a=.1, alpha=-np.pi/2),
                RevoluteDH(a=.1, alpha=0.),
                RevoluteDH(a=.1, alpha=0.)
            ], name="Dummy3DOF"
        )

    def show(self, qs, p):
        self.robot.plot(
            qs, limits=[-0.3, 0.3, -0.3, 0.3, -0.3, 0.3],
            movie="invkine.gif"
        )

    def fkine(self, q):
        return np.array(self.robot.fkine(q))

    def get_position(self, q):
        matrix_t = self.fkine(q)
        xyz = matrix_t[:-1, -1]
        return xyz
