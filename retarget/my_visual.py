from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from const import HAND_VISULIZATION_LINKS

from std_msgs.msg import Float64MultiArray
import rospy
import message_filters
import _thread
import solver

def save_animation(animation: FuncAnimation, path: str, fps: int = 10, dpi: int = 200):
    if path.endswith(".gif"):
        # save the animation to a gif file
        animation.save(path, fps=fps, dpi=dpi, writer="pillow")
    elif path.endswith(".mp4"):
        # save the animation to a mp4 file
        animation.save(path, fps=fps, dpi=dpi, writer="ffmpeg")
    else:
        raise ValueError(f"Unsupported file type: {path}")

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from const import HAND_VISULIZATION_LINKS

from std_msgs.msg import Float64MultiArray
import rospy
import message_filters
import _thread
import solver

def save_animation(animation: FuncAnimation, path: str, fps: int = 10, dpi: int = 200):
    if path.endswith(".gif"):
        # save the animation to a gif file
        animation.save(path, fps=fps, dpi=dpi, writer="pillow")
    elif path.endswith(".mp4"):
        # save the animation to a mp4 file
        animation.save(path, fps=fps, dpi=dpi, writer="ffmpeg")
    else:
        raise ValueError(f"Unsupported file type: {path}")


def plot_two_hands_motion_keypoints(
    keypoints1: np.ndarray, keypoints2: np.ndarray, path: Optional[str] = None
):
    assert keypoints1.ndim == 2

    fig = plt.figure()
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)

    def update():
        ax.clear()

        for line in HAND_VISULIZATION_LINKS:
            ax.plot3D(
                keypoints1[ line, 0],
                keypoints1[line, 1],
                keypoints1[line, 2],
                "gray",
            )
            ax.plot3D(
                keypoints2[line, 0],
                keypoints2[line, 1],
                keypoints2[line, 2],
                "gray",
            )

        ax.scatter3D(
            keypoints1[ :, 0],
            keypoints1[ :, 1],
            keypoints1[ :, 2],
            "black",
        )
        ax.scatter3D(
            keypoints2[ :, 0],
            keypoints2[ :, 1],
            keypoints2[ :, 2],
            "red",
        )
        ax.set_xlim3d(-0.2, 0.2)
        ax.set_ylim3d(-0.2, 0.2)
        ax.set_zlim3d(-0.2, 0.2)
        # ax.set_title(f"Frame {frame:03d}", loc="center")

    # anim = FuncAnimation(fig, update, frames=keypoints1.shape[0], interval=100)
    update()
    print(1)
    plt.show()

    # if path is not None:
    #     save_animation(anim, path)
        

def callback(leap_motion_sub,hand_sub):
    print("msg received!")
    lm_joint=list(leap_motion_sub.data)
    lm_joint=np.array(lm_joint).reshape((21,3))
    hand_data=list(hand_sub.data)
    hand_data=np.array(hand_data)
    hand_joint=solver.get_pose(hand_data)
    plot_two_hands_motion_keypoints(lm_joint,hand_joint)

    
    


if __name__ == "__main__":
    rospy.init_node('hand_visualizer',anonymous=True)

    # rospy.Subscriber('leap_motion_value',Float64MultiArray,callback)
    leap_motion_sub = message_filters.Subscriber('leap_motion_value', Float64MultiArray)# filter
    hand_sub = message_filters.Subscriber('hand_joint_value', Float64MultiArray)#订阅第二个话题，depth图像

    sync = message_filters.ApproximateTimeSynchronizer([leap_motion_sub, hand_sub], 10,1)#同步时间戳，具体参数含义需要查看官方文档。
    sync.registerCallback(callback)#执行反馈函数
    print("ready")
    rospy.spin()