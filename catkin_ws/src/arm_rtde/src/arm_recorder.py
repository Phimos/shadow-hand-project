import copy
import rtde_receive
import rtde_control
import time
import rospy
from arm_rtde.msg import full
from std_msgs.msg import Float64MultiArray
rtde_r = rtde_receive.RTDEReceiveInterface("127.0.0.1")







if __name__=="__main__":
    rospy.init_node("arm_rtde_recorder")
    arm_recorder=rospy.Publisher("arm_rtde_message",full,queue_size=10)
    arm_sender=rospy.Publisher("arm_to_hand",Float64MultiArray,queue_size=100)
    rate=rospy.Rate(10)
    while not rospy.is_shutdown():
        cur_arm_message=full()
        cur_arm_message.timestamp=rtde_r.getTimestamp()
        cur_arm_message.target_joint_position=rtde_r.getTargetQ()
        cur_arm_message.target_joint_velocity=rtde_r.getTargetQd()
        cur_arm_message.target_joint_acceleration=rtde_r.getTargetQdd()
        cur_arm_message.target_joint_moments=rtde_r.getTargetMoment()
        cur_arm_message.actual_joint_position=rtde_r.getActualQ()
        cur_arm_message.actual_joint_velocity=rtde_r.getActualQd()
        cur_arm_message.actual_tcp_pose=rtde_r.getActualTCPPose()
        cur_arm_message.actual_tcp_speed=rtde_r.getActualTCPSpeed()
        cur_arm_message.actual_tcp_force=rtde_r.getActualTCPForce()
        cur_arm_message.target_tcp_pose=rtde_r.getTargetTCPPose()
        cur_arm_message.target_tcp_speed=rtde_r.getTargetTCPSpeed()
        cur_arm_message.target_tcp_accelerometer=rtde_r.getActualToolAccelerometer()
        arm_recorder.publish(cur_arm_message)
        
        arm_joint_mess=Float64MultiArray()
        arm_joint_mess.data=rtde_r.getActualQ()
        arm_sender.publish(arm_joint_mess)
        rate.sleep()
        
    


