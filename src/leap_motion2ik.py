import sys
sys.path.append('/home/wangqx/Leap_Motion/leap-sdk-python3')
import Leap
import numpy as np
import rospy
from std_msgs.msg import Float64MultiArray
import threading
import rtde_control
import rtde_receive
import copy
import time

arm_joint_latest=[0,1,2,3]

def callback(data):
    arm_value=list(data.data)
    print(arm_value)
    pose_new=copy.deepcopy(pose_home)
    pose_new[0]+=arm_value[2]/1000
    pose_new[1]+=arm_value[0]/1000
    pose_new[2]+=(arm_value[1]-100)/1000
    rtde_c.moveL(pose_new,3,3)
    #time.sleep(1)

    #joint_new=rtde_c.getInverseKinematics(pose_new)
    #rtde_c.moveJ(joint_new,3,3,False)
    #print(joint_new)
    #arm_joint_latest=copy.deepcopy(joint_new)
    
    #rospy.sleep(0.1)
    

class thread_publisher(threading.Thread):
    def __init__(self, thread_name):
        threading.Thread.__init__(self,name=thread_name)
    def run(self):
        pub=rospy.Publisher('ik2arm', Float64MultiArray, queue_size=1000)
        rate=rospy.Rate(10)
        while not rospy.is_shutdown():
            mess=Float64MultiArray(data=arm_joint_latest)
            pub.publish(mess)
            rate.sleep()



if __name__=="__main__":
    rtde_c=rtde_control.RTDEControlInterface("192.168.56.1")
    rtde_r=rtde_receive.RTDEReceiveInterface("192.168.56.1")
    pose_home=[-0.5188711681730895, -0.13329963413541965, 
               0.19710102939573393, -1.2217044950771405, -1.221709058801307, 1.201105662901229]
    rtde_c.moveL(pose_home)
    rospy.init_node('arm_value_test', anonymous=True)
    rospy.Subscriber('arm_value2ik',Float64MultiArray,callback)
    print("ready to subscribe")
    send_thread=thread_publisher("send_thread")
    send_thread.start()
    rospy.spin()