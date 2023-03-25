import copy
import rtde_receive
import rtde_control
import time

rtde_c = rtde_control.RTDEControlInterface("127.0.0.1")

joint_target=[0,-1.25,2.00,-0.733,1.5708,-3.1416]
rtde_c.moveJ(joint_target)

pose_home=[-0.5188711681730899, -0.13329963413541968, 0.1971010293957338, -1.2217044950771407, -1.221709058801305, 1.2011056629012318]

while True:
    for i in range (1,10):
        cur_target=copy.deepcopy(pose_home)
        cur_target[0]+=i*0.01
        rtde_c.servoL(cur_target,0.5, 0.1, 0.2, 0.2, 600)
        time.sleep(0.1)
    for i in range (1,10):
        cur_target=copy.deepcopy(pose_home)
        cur_target[0]-=i*0.01
        rtde_c.servoL(cur_target,0.5, 0.1, 0.2, 0.2, 600)
        time.sleep(0.1)
    for i in range (1,10):
        cur_target=copy.deepcopy(pose_home)
        cur_target[1]+=i*0.01
        rtde_c.servoL(cur_target,0.5, 0.1, 0.2, 0.2, 600)
        time.sleep(0.1)
    for i in range (1,10):
        cur_target=copy.deepcopy(pose_home)
        cur_target[1]-=i*0.01
        rtde_c.servoL(cur_target,0.5, 0.1, 0.2, 0.2, 600)
        time.sleep(0.1)
    for i in range (1,10):
        cur_target=copy.deepcopy(pose_home)
        cur_target[2]+=i*0.01
        rtde_c.servoL(cur_target,0.5, 0.1, 0.2, 0.2, 600)
        time.sleep(0.1)
    for i in range (1,10):
        cur_target=copy.deepcopy(pose_home)
        cur_target[2]-=i*0.01
        rtde_c.servoL(cur_target,0.5, 0.1, 0.2, 0.2, 600)
        time.sleep(0.1)


