import sys
sys.path.append('/home/wangqx/Leap_Motion/leap-sdk-python3')
import Leap
import numpy as np
import rospy
from std_msgs.msg import Float64MultiArray
import threading
import time

send=False
armpoints=np.zeros((2,3))

def leap_vector_to_numpy(vector) -> np.ndarray:
    """Converts a Leap Motion `Vector` to a numpy array."""
    return np.array([vector.x, vector.y, vector.z])


def leap_hand_to_keypoints(hand) -> np.ndarray:
    """Converts a Leap Motion `Hand` to a numpy array of keypoints."""
    keypoints = np.zeros((21, 3))

    keypoints[0, :] = leap_vector_to_numpy(hand.wrist_position)

    for finger in hand.fingers:
        finger_index = finger.type
        for bone_index in range(0, 4):
            bone = finger.bone(bone_index)
            index = 1 + finger_index * 4 + bone_index
            keypoints[index, :] = leap_vector_to_numpy(bone.next_joint)
    armpoints[0,:]=leap_vector_to_numpy(hand.palm_position)
    armpoints[1,:]=leap_vector_to_numpy(hand.palm_normal)
    print(armpoints)
    return keypoints


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
            keypoints= leap_hand_to_keypoints(hand)
            
            mess=Float64MultiArray(data=keypoints.ravel())
            pub.publish(mess)
            send=True
            #print(keypoints)
            self.cache.append(keypoints)


class thread_publisher(threading.Thread):
    def __init__(self, thread_name):
        threading.Thread.__init__(self,name=thread_name)
    def run(self):
        pub2 = rospy.Publisher('arm_value2ik', Float64MultiArray, queue_size=2)
        rate=rospy.Rate(1) 

        while not rospy.is_shutdown():
        # while 1:
            mess2=Float64MultiArray(data=armpoints.copy().ravel())
            print("publish")
            pub2.publish(mess2)
            send=False
            # time.sleep(1)
            rate.sleep()







def main():
    # Create a sample listener and controller
    listener = SampleListener()
    controller = Leap.Controller()

    # Have the sample listener receive events from the controller
    controller.add_listener(listener)

    # Keep this process running until Enter is pressed
    print("Press Enter to quit...")
    try:
        sys.stdin.readline()
        np.save("leap_motion.npy", np.array(listener.cache))
    except KeyboardInterrupt:
        pass
    finally:
        # Remove the sample listener when done
        controller.remove_listener(listener)


if __name__ == "__main__":
    rospy.init_node('leap_motion2retarget', anonymous=True)
    global pub
    pub = rospy.Publisher('leap_motion_value', Float64MultiArray, queue_size=1000)
    send_thread=thread_publisher("send_thread")
    send_thread.start()
    
    main()