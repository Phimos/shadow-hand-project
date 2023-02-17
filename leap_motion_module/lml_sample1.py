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

from scipy.spatial.transform import Rotation
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
                
        
def vec_mol(vec):
    sum=0.0
    for num in vec:
            sum=sum+num*num
    return np.sqrt(sum)

def normalized(vector: np.ndarray) -> np.ndarray:
    assert vector.ndim == 1
    return vector / np.linalg.norm(vector)

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
            pose_new=copy.deepcopy(pose_home)
            
            rotation_arm_init = Rotation.from_euler("xyz", pose_new[3:]).as_matrix()
            translation_arm_init = pose_new[:3]
            
            # print("Pose New")
            # print(pose_new)
            
            # # the following is performed in leap_motion coordinate
            
            # home_mat=np.array([[0,-1,0],[0,0,-1],[0,0,0]])
            # des_1=arm_value[1]/vec_mol(arm_value[1])
            
            # des_2=arm_value[2]-arm_value[0]
            # des_2=-des_2/vec_mol(des_2)
            
            # print(des_1,des_2)
            # des_mat=np.append([des_1],[des_2],axis=0)
            # des_mat=np.append(des_mat,[[0,0,0]],axis=0)
            # print(home_mat)
            # print(des_mat)
            # T,R,t=best_fit_transform(home_mat,des_mat)
            # print(R, t)
            
            # q_rot=Rotation.from_matrix(R)
            # eular_leap=q_rot.as_euler('xyz')
            # eular_ur=np.array([eular_leap[0],-eular_leap[1],-eular_leap[2]])
            
            leap_motion_init = np.array(
                [
                    [0, -1, 0],
                    [0, 0, -1],
                    [0, 0, 0]
                ]
            )
            robot_arm_init = np.array(
                [
                    [0, 1, 0],
                    [0, 0, 1],
                    [0, 0, 0]
                ]
            )
            
            palm_norm = normalized(arm_value[1])
            palm_wrist = normalized(arm_value[2] - arm_value[0])
            leap_motion_current = np.array(
                [
                    palm_norm,
                    palm_wrist,
                    [0, 0, 0]
                ]
            )
            
            T0, R0, t0 = best_fit_transform(leap_motion_init, robot_arm_init)
            
            robot_arm_current = (R0 @ leap_motion_current.T).T 
            
            T1, R1, t1 = best_fit_transform(robot_arm_init, robot_arm_current)
            
            
            
            print(pose_home[:3])
            
            
            rotation_new = np.matmul(rotation_arm_init, R1)
            translation_new = np.dot(rotation_arm_init, t1) + translation_arm_init
            
            euler = Rotation.from_matrix(rotation_new).as_euler("xyz")
            
            pose_new = np.concatenate([translation_new, euler])
            # pose_new[:3] = pose_home[:3]
            
            print("pose new")
            print(pose_new[:3])
            
            
            print("diff translation")
            print(translation_new, pose_home[:3])
            
            print("TRt0")
            print(T0, R0, t0)
            print("TRt1")
            print(T1, R1, t1)

            
            # pose_new[0]+=(arm_value[2]-50)/300
            # pose_new[1]+=arm_value[0]/300
            # pose_new[2]+=(arm_value[1]-100)/300
            # pose_new[3]+=eular_ur[0]
            # pose_new[4]+=eular_ur[1]
            # pose_new[5]+=eular_ur[2]
                                 
            print("\n")
            print(pose_new)
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
    print(0)
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
    
    