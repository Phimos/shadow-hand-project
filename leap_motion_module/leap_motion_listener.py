import sys

sys.path.append('/home/wangqx/Leap_Motion/leap-sdk-python3')
import Leap
import numpy as np
import rospy
from std_msgs.msg import Float64MultiArray


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

        print("Frame id %d" % frame.id)
        for hand in frame.hands:
            print("Left hand" if hand.is_left else "Right hand") 
            keypoints= leap_hand_to_keypoints(hand)
            keypoints=keypoints.ravel()
            print(keypoints)
            self.cache.append(keypoints)

# def leap_motion2retarget():
# 	# ROS节点初始化
#     rospy.init_node('leap_motion2retarget', anonymous=True)

# 	# 创建一个Publisher，发布名为/turtle1/cmd_vel的topic，消息类型为geometry_msgs::Twist，队列长度10
#     pub = rospy.Publisher('/leap_motion2retarget', Float64MultiArray, queue_size=10)

# 	#设置循环的频率
#     rate = rospy.Rate(10) 

#     while not rospy.is_shutdown():
# 		# 发布消息
#         # pub.publish()
        
# 		# 按照循环频率延时
#         rate.sleep()
    

if __name__ == '__main__':
    listener = SampleListener()
    controller = Leap.Controller()

    # Have the sample listener receive events from the controller
    controller.add_listener(listener)
    # try:
    #     rospy.init_node('leap_motion2retarget', anonymous=True)

	#     # 创建一个Publisher，发布名为/turtle1/cmd_vel的topic，消息类型为geometry_msgs::Twist，队列长度10
    #     pub = rospy.Publisher('/leap_motion2retarget', Float64MultiArray, queue_size=10)

	#     #设置循环的频率
    #     rate = rospy.Rate(10) 

    #     while not rospy.is_shutdown():
    #         if len(listener.cache)!=0:
	# 	    # 发布消息
    #             pub.publish(Float64MultiArray(listener.cache[-1].ravel()))
	# 	    # 按照循环频率延时
    #         rate.sleep()
    # except rospy.ROSInterruptException:
    #     pass


