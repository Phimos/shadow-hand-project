import glob
import os
import time
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import nlopt
import numpy as np
import torch
from natsort import natsorted
from scipy import signal
from tqdm import tqdm

from align import best_fit_transform
from const import HAND_KEYPOINT_NAMES, HAND_VISULIZATION_LINKS
from poselib.skeleton.skeleton3d import ShadowHandSkeletonState, ShadowHandSkeletonTree
from shadow_hand import ShadowHandModule
from smooth import VelocityFilter
from visualization import (
    plot_hand_keypoints,
    plot_hand_motion_keypoints,
    plot_two_hands_motion_keypoints,
)


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


class RetargetSolver:
    def __init__(
        self,
        task_space_vectors: List[Tuple[str, str]],
        weights: Optional[np.ndarray] = None,
    ):
        self.shadow_hand = ShadowHandModule()
        self.dof = self.shadow_hand.dof
        self.optimizer = self.create_optimizer()
        self.loss_fn = torch.nn.SmoothL1Loss(beta=0.01, reduction="none")
        # self.loss_fn = torch.nn.MSELoss()
        self.indices = torch.tensor(
            [
                (HAND_KEYPOINT_NAMES.index(start), HAND_KEYPOINT_NAMES.index(end))
                for start, end in task_space_vectors
            ],
        )
        self.weights = (
            torch.from_numpy(weights)[..., None] if weights is not None else None
        )

    def create_objective_function(self, target: np.ndarray, latest: np.ndarray):
        target: torch.Tensor = torch.from_numpy(target)
        latest: torch.Tensor = torch.from_numpy(latest)

        # target_tsv = target[self.indices[:, 1]] - target[self.indices[:, 0]]

        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            x: torch.Tensor = torch.from_numpy(x.copy()).float()
            x.requires_grad_(True)

            keypoints = self.shadow_hand.forward(x)

            # tsv = keypoints[self.indices[:, 1]] - keypoints[self.indices[:, 0]]

            # calculate loss
            # loss = self.loss_fn(tsv, target_tsv)
            loss = self.loss_fn(
                keypoints[self.indices[:, 1]], target[self.indices[:, 1]]
            )
            if self.weights is not None:
                loss = torch.sum(loss * self.weights) / torch.sum(self.weights)
            else:
                loss = torch.mean(loss)

            # add regularization
            loss += 1e-3 * torch.sum((x - latest) ** 2)

            # calculate gradient
            loss.backward()
            if grad.size > 0:
                grad[:] = x.grad.numpy()

            return loss.item()

        return objective

    def inequality_constraint(self, finger: str):
        assert finger in ["FF", "MF", "RF", "LF"]
        if finger == "FF":
            distal_index, middle_index = 4, 3
        elif finger == "MF":
            distal_index, middle_index = 8, 7
        elif finger == "RF":
            distal_index, middle_index = 12, 11
        elif finger == "LF":
            distal_index, middle_index = 17, 16

        def constraint(x: np.ndarray, grad: np.ndarray) -> float:
            # distal joint should be less than middle joint
            if grad.size > 0:
                grad[:] = np.zeros(self.dof)
                grad[distal_index] = 1
                grad[middle_index] = -1
            return x[distal_index] - x[middle_index]

        return constraint

    def create_optimizer(self) -> nlopt.opt:
        optimizer = nlopt.opt(nlopt.LD_SLSQP, self.dof)
        min_rad = self.shadow_hand.min_rad.data.clone().detach().numpy()
        max_rad = self.shadow_hand.max_rad.data.clone().detach().numpy()
        optimizer.set_lower_bounds(min_rad)
        optimizer.set_upper_bounds(max_rad)
        optimizer.add_inequality_constraint(self.inequality_constraint("FF"))
        optimizer.add_inequality_constraint(self.inequality_constraint("MF"))
        optimizer.add_inequality_constraint(self.inequality_constraint("RF"))
        optimizer.add_inequality_constraint(self.inequality_constraint("LF"))
        return optimizer

    def solve(self, target: np.ndarray, latest: Optional[np.ndarray] = None):
        latest = latest if latest is not None else np.zeros(self.dof)
        self.optimizer.set_min_objective(self.create_objective_function(target, latest))
        self.optimizer.set_ftol_abs(1e-5)
        x = self.optimizer.optimize(latest)
        return x


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
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    filenames = glob.glob("hand_pose/*joint*.npy")
    filenames = natsorted(filenames)
    target = np.stack([np.load(filename) for filename in filenames])

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
    # weights = np.array([2.0] * 5 + [1.0] * 5)

    skeleton = ShadowHandSkeletonTree.from_mjcf("shadow_hand_full.xml")
    # solver = RetargetSolver(skeleton, links)
    solver = RetargetSolver(links)
    # solver = RetargetSolver(skeleton, links, weights=weights)
    zero_pose = ShadowHandSkeletonState.zero_pose(skeleton)

    pose = ShadowHandSkeletonState.from_angles(skeleton, torch.ones(23) * 0.05)
    print(skeleton._joint_names)
    # print(skeleton.node_names)
    # exit(0)
    zero_keypoints = zero_pose.keypoints

    # for i in range(10):
    #     shadow = ShadowHandModule()
    #     result = shadow.forward(torch.ones(23) * 0.05)
    #     print(i)
    #     print(result)
    # print(pose.keypoints)
    # exit(0)

    scale_factor = np.mean(
        [calc_scale_factor(target[i], zero_keypoints) for i in range(target.shape[0])]
    )
    target *= scale_factor
    print("scale factor", scale_factor)

    for i in range(target.shape[0]):
        _, R, t = best_fit_transform(
            target[i, [0, 5, 9, 13]],
            zero_keypoints[[0, 5, 9, 13]].detach().numpy(),
        )
        # print(R.shape, t.shape, target[i].shape)
        target[i] = (R @ target[i].T).T + t
        target[i] += zero_keypoints[[5, 9, 13]].detach().numpy().mean(axis=0) - target[
            i, [5, 9, 13]
        ].mean(axis=0)

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
    # plot_hand_motion_keypoints(target, "target_glove_1.gif")
    # exit(0)

    result = np.zeros_like(target)
    latset = np.zeros(solver.dof)
    start = time.time()
    for i in tqdm(range(target.shape[0])):
        latest = solver.solve(target[i], latset)
        result[i] = ShadowHandSkeletonState.from_angles(
            skeleton, torch.from_numpy(latest).float()
        ).keypoints
        print(latest)
    end = time.time()
    print(f"solve {target.shape[0]} frames in {end - start:.2f} seconds")
    print(f"average {target.shape[0] / (end - start):.2f} fps")

    # plot_hand_motion_keypoints(result, "result_glove_1217.gif")
    # plot_hand_motion_keypoints(result)
    # # plot_hand_motion_keypoints(target, result)
    # # plot_two_hands_motion_keypoints(target, result, "result_glove_both_1217.gif")
    # plot_two_hands_motion_keypoints(target, result)
