#!/usr/bin/env /home/yunchong/anaconda3/envs/robot/bin/python
import argparse
import sys

sys.path.append("/home/yunchong/Documents/leap-sdk-python3")

import Leap
import numpy as np
import rospy
import rtde_control
import rtde_receive
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Float64MultiArray

ARM_HOME_POSITION = [
    -0.7321133556947282,
    -0.17414999999104813,
    0.2569021173596485,
    -1.1830891591035162,
    -1.1830891595975745,
    1.2257274537352985,
]
ARM_HOME_JOINT_POSITION = [0, -1.25, 2.00, -np.pi / 4, np.pi / 2, -np.pi]

LM_HOME_WRIST_POSITION = np.array([30.0, 160.0, 40.0])
LM_HOME_PALM_NORMAL = np.array([0.0, -1.0, 0.0])
LM_HOME_DIRECTION = np.array([0.0, 0.0, -1.0])

LM_TO_UR_ROTATION = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])


def leap_vector_to_numpy(vector) -> np.ndarray:
    """Converts a Leap Motion `Vector` to a numpy array."""
    return np.array([vector.x, vector.y, vector.z])


def best_fit_transform(A, B):
    """
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    """

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # homogeneous transformation
    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


class LeapListener(Leap.Listener):
    finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
    bone_names = ["Metacarpal", "Proximal", "Intermediate", "Distal"]
    state_names = ["STATE_INVALID", "STATE_START", "STATE_UPDATE", "STATE_END"]

    def __init__(self, rtde_c, rtde_r, scale=2.5) -> None:
        super().__init__()
        self.rtde_c = rtde_c
        self.rtde_r = rtde_r
        self.scale = scale / 1000
        self.cache = []

    def on_init(self, controller):
        print("Initialized")

    def on_connect(self, controller):
        print("Connected")

    def on_disconnect(self, controller):
        # Note: not dispatched when running in a debugger.
        print("Disconnected")

    def on_exit(self, controller):
        print("Exited")

    def get_right_hand(self, frame):
        for hand in frame.hands:
            if hand.is_right:
                return hand
        return None

    def on_frame(self, controller):
        # Get the most recent frame and report some basic information
        frame = controller.frame()

        right_hand = self.get_right_hand(frame)

        if right_hand is None:
            return

        # end-effector position
        print("end-effector position: %s" % self.rtde_r.getActualTCPPose())

        current_position = self.rtde_r.getActualTCPPose()
        current_position = np.array(current_position)
        print("current_position: %s" % current_position)
        print(type(current_position))

        print("Frame id %d" % frame.id)
        print("Frame timestamp: %d" % frame.timestamp)
        print("Wrist position: %s" % leap_vector_to_numpy(right_hand.wrist_position))
        print("Palm normal: %s" % leap_vector_to_numpy(right_hand.palm_normal))
        print("Direction: %s" % leap_vector_to_numpy(right_hand.direction))

        # init position
        init_wrist_position = np.copy(LM_HOME_WRIST_POSITION)
        init_palm_normal = np.copy(LM_HOME_PALM_NORMAL)
        init_direction = np.copy(LM_HOME_DIRECTION)
        init_wrist_position *= self.scale
        init_palm_normal *= 0.2 * self.scale
        init_direction *= 0.2 * self.scale

        # fetch data from leap motion
        wrist_position = leap_vector_to_numpy(right_hand.wrist_position)
        palm_normal = leap_vector_to_numpy(right_hand.palm_normal)
        direction = leap_vector_to_numpy(right_hand.direction)
        wrist_position *= self.scale
        palm_normal *= 0.2 * self.scale
        direction *= 0.2 * self.scale

        # convert to ur coordinate
        init_wrist_position = np.dot(LM_TO_UR_ROTATION, init_wrist_position)
        init_palm_normal = np.dot(LM_TO_UR_ROTATION, init_palm_normal)
        init_direction = np.dot(LM_TO_UR_ROTATION, init_direction)

        wrist_position = np.dot(LM_TO_UR_ROTATION, wrist_position)
        palm_normal = np.dot(LM_TO_UR_ROTATION, palm_normal)
        direction = np.dot(LM_TO_UR_ROTATION, direction)

        relative_wrist_position = wrist_position - init_wrist_position

        init_points = np.array([np.zeros(3), init_palm_normal, init_direction])

        point_wrist = relative_wrist_position.copy()
        point_palm_normal = relative_wrist_position + palm_normal
        point_direction = relative_wrist_position + direction
        points = np.array([point_wrist, point_palm_normal, point_direction])

        _, rotation, translation = best_fit_transform(init_points, points)

        print("rotation: %s" % rotation)
        print("translation: %s" % translation)

        ur_home_position = np.array(ARM_HOME_POSITION)

        # convert rotation vector to rotation matrix
        init_rotation = R.from_rotvec(ur_home_position[3:]).as_matrix()
        init_translation = ur_home_position[:3]

        # create composed rotation and translation
        composed_rotation = np.dot(init_rotation, rotation)
        composed_translation = np.dot(init_rotation, translation) + init_translation

        # convert rotation matrix to rotation vector
        composed_rotvec = R.from_matrix(composed_rotation).as_rotvec()

        # create new 6d position x, y, z, rx, ry, rz
        current_position = np.zeros(6)
        current_position[:3] = composed_translation
        current_position[3:] = composed_rotvec

        print("new current_position: %s" % current_position)

        rtde_c.servoL(current_position.tolist(), 0.5, 0.1, 0.2, 0.2, 600)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="127.0.0.1", help="IP of the robot")
    args = parser.parse_args()

    rospy.init_node("robot_arm_controller", anonymous=True)

    print("robot arm controller node start ...")

    rtde_c = rtde_control.RTDEControlInterface(args.ip)
    rtde_r = rtde_receive.RTDEReceiveInterface(args.ip)

    print("moving to home position ...")
    rtde_c.moveJ(ARM_HOME_JOINT_POSITION, 0.5, 0.5)

    print("home position: %s" % rtde_r.getActualTCPPose())

    listener = LeapListener(rtde_c, rtde_r)
    controller = Leap.Controller()

    controller.add_listener(listener)

    print("Press Enter to quit...")
    try:
        sys.stdin.readline()
    except KeyboardInterrupt:
        pass
    finally:
        # Remove the sample listener when done
        controller.remove_listener(listener)
