import rtde_control
import rtde_receive

rtde_c=rtde_control.RTDEControlInterface("192.168.56.1")
rtde_r=rtde_receive.RTDEReceiveInterface("192.168.56.1")
pose_home=[-0.5188711681730895, -0.13329963413541965, 
               0.19710102939573393, -1.2217044950771405, -1.221709058801307, 1.201105662901229]
joint_target=[0,-1.25,2.00,-0.733,1.5708,-3.1416]
rtde_c.moveJ(joint_target)