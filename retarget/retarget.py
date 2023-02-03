import glob
import os
import time
from typing import List, Optional, Tuple

import numpy as np
from align import best_fit_transform
from const import HAND_KEYPOINT_NAMES, HAND_VISULIZATION_LINKS
from natsort import natsorted
from scipy import signal
from smooth import VelocityFilter
from solver import MotionMapper
from tqdm import tqdm
from visualization import (
    plot_hand_keypoints,
    plot_hand_motion_keypoints,
    plot_two_hands_motion_keypoints,
)


def normalized(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def extend_pinky(keypoints: np.ndarray) -> np.ndarray:
    """Extends the pinky finger to the length of mean of the other three fingers."""
    assert keypoints.shape == (21, 3)
    pinky = keypoints[17:21, :].copy()

    index = keypoints[5:9, :]
    middle = keypoints[9:13, :]
    ring = keypoints[13:17, :]

    fingers = np.stack([index, middle, ring], axis=0)

    proximal = np.mean(np.linalg.norm(fingers[:, 1, :] - fingers[:, 0, :], axis=1))
    intermediate = np.mean(np.linalg.norm(fingers[:, 2, :] - fingers[:, 1, :], axis=1))
    distal = np.mean(np.linalg.norm(fingers[:, 3, :] - fingers[:, 2, :], axis=1))

    proximal_direction = normalized(pinky[1, :] - pinky[0, :])
    intermediate_direction = normalized(pinky[2, :] - pinky[1, :])
    distal_direction = normalized(pinky[3, :] - pinky[2, :])

    pinky[1, :] = pinky[0, :] + proximal * proximal_direction
    pinky[2, :] = pinky[1, :] + intermediate * intermediate_direction
    pinky[3, :] = pinky[2, :] + distal * distal_direction

    keypoints = np.concatenate([keypoints[:17, :].copy(), pinky], axis=0)
    return keypoints


def filter_position_sequence(position_seq: np.ndarray, wn=5, fs=25):
    sos = signal.butter(2, wn, "lowpass", fs=fs, output="sos", analog=False)
    seq_shape = position_seq.shape
    if len(seq_shape) < 2:
        raise ValueError(
            f"Joint Sequence must have data with 3-dimension or 2-dimension, but got shape {seq_shape}"
        )
    result_seq = np.empty_like(position_seq)
    if len(seq_shape) == 3:
        for i in range(seq_shape[1]):
            for k in range(seq_shape[2]):
                result_seq[:, i, k] = signal.sosfilt(sos, position_seq[:, i, k])
    elif len(seq_shape) == 2:
        for i in range(seq_shape[1]):
            result_seq[:, i] = signal.sosfilt(sos, position_seq[:, i])

    return result_seq


def calc_link_lengths(keypoints: np.ndarray, links: List[Tuple[int, int]]):
    link_lengths = []
    for start, end in links:
        link_lengths.append(np.linalg.norm(keypoints[start] - keypoints[end]))
    return np.array(link_lengths)


def calc_scale_factor(source: np.ndarray, target: np.ndarray) -> float:
    source_lengths = calc_link_lengths(source, HAND_VISULIZATION_LINKS)
    target_lengths = calc_link_lengths(target, HAND_VISULIZATION_LINKS)
    return np.sum(target_lengths) / np.sum(source_lengths)


if __name__ == "__main__":
    filenames = glob.glob("hand_pose/*joint*.npy")
    filenames = natsorted(filenames)
    target = np.stack([np.load(filename) for filename in filenames])

    target = np.load("leap_motion.npy")[::4]

    # target = np.load(
    #     "/Users/pims/Downloads/MatlabVisualizeGloveData-master/data/HandData_apple_VR_5.npy"
    # )
    # target = target[::4]

    target = target - target[:, 0:1, :]

    # plot_hand_motion_keypoints(target)

    links = [
        ("palm", "thmiddle"),
        ("palm", "ffmiddle"),
        ("palm", "mfmiddle"),
        ("palm", "rfmiddle"),
        ("palm", "lfmiddle"),
        ("palm", "thtip"),
        ("palm", "fftip"),
        ("palm", "mftip"),
        ("palm", "rftip"),
        ("palm", "lftip"),
    ]
    solver = MotionMapper()
    zero_keypoints = solver.get_zero_pose()

    scale_factor = np.mean(
        [calc_scale_factor(target[i], zero_keypoints) for i in range(target.shape[0])]
    )
    target *= scale_factor
    print("scale factor", scale_factor)

    for i in range(target.shape[0]):
        _, R, t = best_fit_transform(
            target[i, [0, 5, 9, 13]],
            zero_keypoints[[0, 5, 9, 13]],
        )
        # print(R.shape, t.shape, target[i].shape)
        target[i] = (R @ target[i].T).T + t
        target[i] += zero_keypoints[[5, 9, 13]].mean(axis=0) - target[
            i, [5, 9, 13]
        ].mean(axis=0)

    plot_hand_motion_keypoints(target)

    # human_link_lengths = calc_link_lengths(target[0], HAND_VISULIZATION_LINKS)
    # robot_link_lengths = calc_link_lengths(zero_pose.keypoints, HAND_VISULIZATION_LINKS)
    # for i, (start, end) in enumerate(HAND_VISULIZATION_LINKS):
    #     print(start, end, robot_link_lengths[i] / human_link_lengths[i])
    # print(calc_link_lengths(target[0], HAND_VISULIZATION_LINKS))
    # print(calc_link_lengths(zero_pose.keypoints, HAND_VISULIZATION_LINKS))
    # exit(0)
    # plot_hand_motion_keypoints(target, "target.gif")
    # target = filter_position_sequence(target)
    # plot_hand_motion_keypoints(target, "default_filter.gif")
    # exit(0)

    velocity_filter = VelocityFilter(5, 5)
    for i in range(target.shape[0]):
        target[i] = velocity_filter(target[i])
        target[i] = extend_pinky(target[i])
    # plot_hand_motion_keypoints(target, "target_glove_1.gif")
    # exit(0)

    result = np.zeros_like(target)
    latset = np.zeros(solver.dof)
    start = time.time()
    for i in tqdm(range(target.shape[0])):
        latest = solver.step(target[i])
        # latest = solver.solve(target[i], latset)
        result[i] = solver.get_pose(latest)
    end = time.time()
    print(f"solve {target.shape[0]} frames in {end - start:.2f} seconds")
    print(f"average {target.shape[0] / (end - start):.2f} fps")

    # plot_hand_motion_keypoints(result, "result_glove_1217.gif")
    plot_hand_motion_keypoints(result)
    # # plot_hand_motion_keypoints(target, result)
    # # plot_two_hands_motion_keypoints(target, result, "result_glove_both_1217.gif")
    plot_two_hands_motion_keypoints(target, result)
