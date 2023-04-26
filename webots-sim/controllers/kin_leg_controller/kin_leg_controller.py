"""simple-kinematic-leg controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
import numpy as np
from controller import Supervisor  # type: ignore


L1 = 0.15
L2 = 0.15


class KinLeg:

    def __init__(self):
        # create the Robot instance.
        self.robot = Supervisor()

        # get joints references
        self.joints = [self.robot.getDevice(f"joint{i}") for i in range(1, 4)]

        # get the time step of the current world.
        self.timestep = int(self.robot.getBasicTimeStep())

    def delay(self, ms):
        iter = ms // self.timestep
        while iter > 0:
            iter -= 1
            self.step()

    def setJoints(self, q):
        for i in range(3):
            self.joints[i].setPosition(q[i])
        self.delay(50)

    def invkine(self, x, y, z) -> list[float]:
        # params: x, y, z (target position)
        # returns: [q1, q2, q3] (joint angles)
        L = np.sqrt(x*x + y*y + z*z)
        L1L = (L1*L1 + L*L - L2*L2) / (2*L1*L)
        L1L2 = (L1*L1 + L2*L2 - L*L) / (2*L1*L2)
        return np.array([
            -np.arctan2(x, y),
            0.5*np.pi - np.arccos(L1L) - np.arctan2(np.sqrt(x*x + y*y), z),
            0.5*np.pi - np.arccos(L1L2)
        ])

    def step(self):
        return self.robot.step(self.timestep)


def ball_position(ball_pos):
    p = T10 @ ball_pos
    return p[:, -1]


if __name__ == "__main__":

    kin_leg = KinLeg()

    ball = kin_leg.robot.getFromDef("BALL")
    T01 = np.array(
        kin_leg.robot.getFromDef("JOINT1").getPose()
    ).reshape(4, 4)
    T10 = np.linalg.inv(T01)

    while kin_leg.step() != -1:
        ball_pos = ball.getField("translation").getSFVec3f()
        ball_pos = np.array(
            ball_pos + [1.]
        ).reshape(4, 1)
        ball_pos = ball_position(ball_pos)

        q = kin_leg.invkine(*ball_pos[:-1])

        kin_leg.setJoints(q)
