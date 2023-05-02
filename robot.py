import numpy as np


class RobotPuma560:
    def __init__(self, dh_t) -> None:
        self.dh_table = dh_t

    def dh_transform(self, theta, alpha, r, d):
        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        T = np.array(
            [
                [ct, -st * ca, st * sa, r * ct],
                [st, ct * ca, -ct * sa, r * st],
                [0, sa, ca, d],
                [0, 0, 0, 1],
            ]
        )
        return T

    def fkine(self, theta, alpha, r, d, offset=np.array([0.0])):  # , 0, 0, 0, 0])):
        
        # print(f"Thetas = {theta}")
        # print(f"alphas = {alpha}")
        # print(f"rs = {r}")
        # print(f"ds = {d}\n")

        T = np.eye(4)
        theta = theta + offset

        for i in range(len(theta)):
            T_i = self.dh_transform(theta[i], alpha[i], r[i], d[i])
            T = np.dot(T, T_i)

        return T

    def get_position(self, thetas):
        new_dh_table = [thetas] + [self.dh_table[:, i] for i in range(1, 4)]
        matrix_t = self.fkine(*new_dh_table)

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
        roll = np.arctan2(r10 / np.cos(pitch), r00 / np.cos(pitch))
        yaw = np.arctan2(r21 / np.cos(pitch), r22 / np.cos(pitch))

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
