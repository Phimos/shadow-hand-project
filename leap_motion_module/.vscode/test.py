import copy
import rtde_control
import rtde_receive
import time

rtde_c=rtde_control.RTDEControlInterface("192.168.56.1")
rtde_r=rtde_receive.RTDEReceiveInterface("192.168.56.1")

joint_target=[0,-1.25,2.00,-0.733,1.5708,-3.1416]
rtde_c.moveJ(joint_target)

pose_home=rtde_r.getActualTCPPose()
cur=copy.deepcopy(pose_home)
speed=[0,0,0.05,0,0,0]
v=0.1
for i in range(100):
    rtde_c.speedL(speed,0.1,1)
    time.sleep(0.1)
    speed[2]+=0.1
rtde_c.speedStop()