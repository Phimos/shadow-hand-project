import os
import sys
sys.path.append('/home/wangqx/Leap_Motion/leap-sdk-python3')
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
send=False
armpoints=np.array([[0,100,0],[0.01,-1,0.01],[0.01,0.01,1]])
switch=True

def leap_vector_to_numpy(vector) -> np.ndarray:
    """Converts a Leap Motion `Vector` to a numpy array."""
    return np.array([vector.x, vector.y, vector.z])


def leap_hand_to_keypoints(hand) -> np.ndarray:
    """Converts a Leap Motion `Hand` to a numpy array of keypoints."""
    #print(hand.palm_position) 
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
    armpoints[2,:]=leap_vector_to_numpy(hand.wrist_position)
    return keypoints

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

class thread_watch(threading.Thread):
    def __init__(self, thread_name):
        threading.Thread.__init__(self,name=thread_name)
    def run(self):
        watch()
                
        


class thread_publisher(threading.Thread):
    def __init__(self, thread_name):
        threading.Thread.__init__(self,name=thread_name)
    def arcos(self,a,b):
        a1=(b[0]*b[0]+b[1]*b[1])*(a[0]*a[0]+a[1]*a[1])
        a1=np.sqrt(a1)
        a2=(a[0]*b[0]+a[1]*b[1])/a1
        return np.arccos(a2)
    def run(self):
        while 1:
            arm_value=copy.deepcopy(armpoints)
            arm_value=arm_value.ravel()
            pose_new=copy.deepcopy(pose_home)
            rvec=np.array([0,-1,0])
            nvec=arm_value[3:]
            #print(rvec[1:])
            #print(nvec[1:])
            x=self.arcos(rvec[1:],nvec[1:])
            # y=self.arcos(rvec[0,2],nvec[0,2])
            z=self.arcos(rvec[:2],nvec[:2])
            rotx=z*np.sign(nvec[0])
            # roty=x*np.sign(nvec[2])
            #pose_new[3]-=rotx
            #pose_new[4]-=roty
            #print(arm_value)
            # pose=rtde_r.getActualTCPPose()
            
            
            # angle bias
            alpha=0.75
            beta=-1
            gamma=0.3
            theta=1.5
            sigma=0.55
            
            # 在leapmotion的坐标系下计算（palm_position-wristposition）的夹角
            
            # x 俯仰 y平面左右 z(rotx) 翻掌
            
            pw_vec=armpoints[0]-armpoints[2]
            pw_x=self.arcos(pw_vec[1:],[0,-1])*np.sign(pw_vec[1])
            pw_y=self.arcos([pw_vec[0],pw_vec[2]],[0,-1])*np.sign(pw_vec[0])-pw_x*alpha+rotx*theta
            pw_x+=rotx*beta+gamma*pw_y
            # 转换到ur坐标
            pw_rotz=pw_y
            pw_roty=pw_x
            
            pose_new[3]+=rotx+sigma*pw_x
            pose_new[4]+=pw_roty
            pose_new[5]-=pw_rotz
            
            
            pose_new[0]+=(arm_value[2]-50)/300
            pose_new[1]+=arm_value[0]/300
            pose_new[2]+=(arm_value[1]-100)/300
            #print(pose_new)
            if switch:
                rtde_c.servoL(pose_new, 0.5, 0.1, 0.2, 0.2, 600)
            else:
                rtde_c.servoStop()
            #print("rotx:",rotx," roty:",roty," rotz:",0)
            #rtde_c.moveJ_IK(pose_new,1,1,True)
            time.sleep(0.1)
            
        







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
        np.save("leap_motion.npy", np.array(listener.cache))
    except KeyboardInterrupt:
        pass
    finally:
        # Remove the sample listener when done
        controller.remove_listener(listener)


if __name__ == "__main__":
    rospy.init_node('leap_motion2retarget', anonymous=True)
    global pub
    print(2)
    rtde_c=rtde_control.RTDEControlInterface("192.168.56.1")
    rtde_r=rtde_receive.RTDEReceiveInterface("192.168.56.1")
    print(3)
    pose_home=[-0.5188711681730895, -0.13329963413541965, 
               0.19710102939573393, -1.2217044950771405, -1.221709058801307, 1.201105662901229]
    joint_target=[0,-1.25,2.00,-0.733,1.5708,-3.1416]
    rtde_c.moveJ(joint_target)
    # rtde_c.moveL(pose_home)
    pub = rospy.Publisher('leap_motion_value', Float64MultiArray, queue_size=1000)
    send_thread=thread_publisher("send_thread")
    send_thread.start()
    watch_thread=thread_watch("watch_thread")
    watch_thread.start()
    print(1)
    while 1:
        a=input()
        if a=="unlock":
            switch=True
        elif a=="exit":
            os.kill()
        else: 
            switch=False
    # main()