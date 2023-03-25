import os
import sys

sys.path.append(
    "/home/user/Documents/LeapDeveloperKit_2.3.1+31549_linux/leap-sdk-python3"
)
import Leap
import numpy as np
import rospy
from std_msgs.msg import Float64MultiArray
import threading
import time
import rtde_control
import rtde_receive
import copy
import cmath
from leap_motion.msg import leap_message

send = False
switch = True


def leap_vector_to_numpy(vector) -> np.ndarray:
    """Converts a Leap Motion `Vector` to a numpy array."""
    return np.array([vector.x, vector.y, vector.z])


def leap_hand_to_keypoints(hand) -> np.ndarray:
    """Converts a Leap Motion `Hand` to a numpy array of keypoints."""
    # print(hand.palm_position)
    keypoints = np.zeros((21, 3))
    armpoints = np.zeros((4, 3))
    keypoints[0, :] = leap_vector_to_numpy(hand.wrist_position)

    for finger in hand.fingers:
        finger_index = finger.type
        for bone_index in range(0, 4):
            bone = finger.bone(bone_index)
            index = 1 + finger_index * 4 + bone_index
            keypoints[index, :] = leap_vector_to_numpy(bone.next_joint)
    armpoints[0, :] = leap_vector_to_numpy(hand.direction)
    armpoints[1, :] = leap_vector_to_numpy(hand.palm_normal)
    armpoints[2, :] = leap_vector_to_numpy(hand.wrist_position)
    armpoints[3, :] = leap_vector_to_numpy(hand.palm_position)
    return keypoints, armpoints


class SampleListener(Leap.Listener):
    finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
    bone_names = ["Metacarpal", "Proximal", "Intermediate", "Distal"]
    state_names = ["STATE_INVALID", "STATE_START", "STATE_UPDATE", "STATE_END"]

    def __init__(self) -> None:
        super().__init__()
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

    def on_frame(self, controller):
        # Get the most recent frame and report some basic information
        frame = controller.frame()

        # print("Frame id %d" % frame.id)
        for hand in frame.hands:
            # print("Left hand" if hand.is_left else "Right hand")
            keypoints, armpoints = leap_hand_to_keypoints(hand)

            message = leap_message()
            message.finger_joint = keypoints.ravel()
            message.direction = armpoints[0]
            message.palm_normal = armpoints[1]
            message.wrist_position = armpoints[2]
            pub.publish(message)
            arm_message = Float64MultiArray()
            arm_message.data = armpoints.ravel()
            pub_arm.publish(arm_message)
            hand_message = Float64MultiArray()
            hand_message.data = keypoints.ravel()
            pub_hand.publish(hand_message)

            print(armpoints)


class thread_watch(threading.Thread):
    def __init__(self, thread_name):
        threading.Thread.__init__(self, name=thread_name)

    def run(self):
        watch()


def watch():
    # Create a sample listener and controller
    listener = SampleListener()
    controller = Leap.Controller()

    # Have the sample listener receive events from the controller
    controller.add_listener(listener)

    # Keep this process running until Enter is pressed
    print("Press Enter to quit...")
    try:
        sys.stdin.readline()

    except KeyboardInterrupt:
        pass
    finally:
        # Remove the sample listener when done
        controller.remove_listener(listener)


if __name__ == "__main__":
    rospy.init_node("leap_motion2retarget", anonymous=True)
    global pub
    # rtde_c=rtde_control.RTDEControlInterface("192.168.56.1")
    # rtde_r=rtde_receive.RTDEReceiveInterface("192.168.56.1")
    # pose_home=[-0.5188711681730895, -0.13329963413541965,
    #            0.19710102939573393, -1.2217044950771405, -1.221709058801307, 1.201105662901229]
    # joint_target=[0,-1.25,2.00,-0.733,1.5708,-3.1416]
    # rtde_c.moveJ(joint_target)
    # rtde_c.moveL(pose_home)
    pub = rospy.Publisher("leap_motion_value", leap_message, queue_size=1000)
    pub_arm = rospy.Publisher("leap_arm", Float64MultiArray, queue_size=100)
    pub_hand = rospy.Publisher("leap_hand", Float64MultiArray, queue_size=100)
    watch_thread = thread_watch("watch_thread")
    watch_thread.start()
