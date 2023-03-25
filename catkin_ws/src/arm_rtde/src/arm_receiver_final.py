import copy
import os
import sys
import numpy as np
import rospy
import rtde_control
import rtde_receive
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Float64MultiArray
import threading

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
switch=True


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



class thread_publisher(threading.Thread):
    def __init__(self, thread_name):
        threading.Thread.__init__(self,name=thread_name)
    def run(self):
        rospy.Subscriber("leap_arm",Float64MultiArray,arm_call_back)
        rospy.spin()
        

def arcos(a,b):
        a1=(b[0]*b[0]+b[1]*b[1])*(a[0]*a[0]+a[1]*a[1])
        a1=np.sqrt(a1)
        a2=(a[0]*b[0]+a[1]*b[1])/a1
        return np.arccos(a2)


def arm_call_back(msg):
            # print("receive")
            pose_home=[-0.5188711681730895, -0.13329963413541965, 
               0.69710102939573393, -1.2217044950771405, -1.221709058801307, 1.201105662901229]
            armpoints_=np.array(msg.data).reshape(4,3)
            armpoints=np.zeros((3,3))
            armpoints[0]=armpoints_[3]
            armpoints[1]=armpoints_[1]
            armpoints[2]=armpoints_[2]
            arm_value=copy.deepcopy(armpoints)
            arm_value=arm_value.ravel()
            pose_new=copy.deepcopy(pose_home)
            rvec=np.array([0,-1,0])
            nvec=arm_value[3:]
            x=arcos(rvec[1:],nvec[1:])
            z=arcos(rvec[:2],nvec[:2])
            rotx=z*np.sign(nvec[0])
            
            
            # angle bias
            alpha=0.75
            beta=-1
            gamma=0.3
            theta=1.5
            sigma=0.55
            
            
            pw_vec=armpoints[0]-armpoints[2]
            pw_x=arcos(pw_vec[1:],[0,-1])*np.sign(pw_vec[1])
            pw_y=arcos([pw_vec[0],pw_vec[2]],[0,-1])*np.sign(pw_vec[0])-pw_x*alpha+rotx*theta
            pw_x+=rotx*beta+gamma*pw_y
            # 转换到ur坐标
            pw_rotz=pw_y
            pw_roty=pw_x
            
            pose_new[3]+=rotx+sigma*pw_x
            pose_new[4]+=pw_roty
            pose_new[5]-=pw_rotz
            
            
            pose_new[0]+=(arm_value[2]-50)/300
            pose_new[1]+=arm_value[0]/300
            pose_new[2]+=(arm_value[1]-210)/300
            if switch:
                rtde_c.servoL(pose_new, 0.5, 0.1, 0.2, 0.2, 600)
            else:
                rtde_c.servoStop()


    
    



if __name__=="__main__":
    rospy.init_node("arm_receriver")
    rtde_c = rtde_control.RTDEControlInterface("192.168.56.1")
    pose_home=[-0.5188711681730899, -0.13329963413541968, 0.6971010293957338, -1.2217044950771407, -1.221709058801305, 1.2011056629012318]
    rtde_c.moveL(pose_home)
    arm_control_thread=thread_publisher("arm_control_thread")
    arm_control_thread.start()
    
    while 1:
        a=input()
        if a=="unlock":
            switch=True
        elif a=="exit":
            sys.exit(0)
        else:
            switch=False
