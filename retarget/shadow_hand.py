from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


@torch.jit.script
def hat(x: torch.Tensor) -> torch.Tensor:
    x0, x1, x2 = x[..., 0], x[..., 1], x[..., 2]
    zeros = torch.zeros_like(x0)
    return torch.stack(
        [
            torch.stack([zeros, -x2, x1], dim=-1),
            torch.stack([x2, zeros, -x0], dim=-1),
            torch.stack([-x1, x0, zeros], dim=-1),
        ],
        dim=-2,
    )


@torch.jit.script
def rotation_matrix_from_angle_axis(
    angle: torch.Tensor, axis: torch.Tensor
) -> torch.Tensor:
    assert angle.device == axis.device
    sin, cos = torch.sin(angle), torch.cos(angle)
    sin, cos = sin[..., None, None], cos[..., None, None]
    omega_hat = hat(axis)
    eye = torch.eye(3, device=angle.device, dtype=torch.float32)
    return eye + sin * omega_hat + (1 - cos) * torch.matmul(omega_hat, omega_hat)


@torch.jit.script
def multiply_transform(
    transform1: Tuple[torch.Tensor, torch.Tensor],
    transform2: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    R1, t1 = transform1
    R2, t2 = transform2
    return torch.matmul(R1, R2), torch.mv(R1, t2) + t1


class Joint:
    name: str
    parent: int
    axis: List[float]
    min_rad: float
    max_rad: float

    def __init__(
        self, name: str, parent: int, axis: List[float], min_rad: float, max_rad: float
    ) -> None:
        self.name = name
        self.parent = parent
        self.axis = axis
        self.min_rad = min_rad
        self.max_rad = max_rad


class Node:
    name: str
    parent: int
    translation: List[float]

    def __init__(self, name: str, parent: int, translation: List[float]) -> None:
        self.name = name
        self.parent = parent
        self.translation = translation


def angle_tensor_to_dict(angles: torch.Tensor | np.ndarray) -> Dict[str, float]:
    assert isinstance(angles, torch.Tensor) or isinstance(angles, np.ndarray)
    if isinstance(angles, torch.Tensor):
        angles = angles.detach().cpu().numpy()
    assert angles.shape == (24,)

    return {
        "rh_WRJ2": angles[0],
        "rh_WRJ1": angles[1],
        "rh_FFJ4": angles[2],
        "rh_FFJ3": angles[3],
        "rh_FFJ2": angles[4],
        "rh_FFJ1": angles[5],
        "rh_MFJ4": angles[6],
        "rh_MFJ3": angles[7],
        "rh_MFJ2": angles[8],
        "rh_MFJ1": angles[9],
        "rh_RFJ4": angles[10],
        "rh_RFJ3": angles[11],
        "rh_RFJ2": angles[12],
        "rh_RFJ1": angles[13],
        "rh_LFJ5": angles[14],
        "rh_LFJ4": angles[15],
        "rh_LFJ3": angles[16],
        "rh_LFJ2": angles[17],
        "rh_LFJ1": angles[18],
        "rh_THJ5": angles[19],
        "rh_THJ4": angles[20],
        "rh_THJ3": angles[21],
        "rh_THJ2": angles[22],
        "rh_THJ1": angles[23],
    }


class ShadowHandModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        nodes: List[Node] = []
        joints: List[Joint] = []

        nodes.append(Node("wrist", -1, [0.0000, 0.0000, 0.0000]))
        nodes.append(Node("palm", 0, [0.0000, 0.0000, 0.0340]))
        nodes.append(Node("ffknuckle", 1, [0.0330, 0.0000, 0.0950]))
        nodes.append(Node("ffproximal", 2, [0.0000, 0.0000, 0.0000]))
        nodes.append(Node("ffmiddle", 3, [0.0000, 0.0000, 0.0450]))
        nodes.append(Node("ffdistal", 4, [0.0000, 0.0000, 0.0250]))
        nodes.append(Node("fftip", 5, [0.0000, 0.0000, 0.0260]))
        nodes.append(Node("mfknuckle", 1, [0.0110, 0.0000, 0.0990]))
        nodes.append(Node("mfproximal", 7, [0.0000, 0.0000, 0.0000]))
        nodes.append(Node("mfmiddle", 8, [0.0000, 0.0000, 0.0450]))
        nodes.append(Node("mfdistal", 9, [0.0000, 0.0000, 0.0250]))
        nodes.append(Node("mftip", 10, [0.0000, 0.0000, 0.0260]))
        nodes.append(Node("rfknuckle", 1, [-0.0110, 0.0000, 0.0950]))
        nodes.append(Node("rfproximal", 12, [0.0000, 0.0000, 0.0000]))
        nodes.append(Node("rfmiddle", 13, [0.0000, 0.0000, 0.0450]))
        nodes.append(Node("rfdistal", 14, [0.0000, 0.0000, 0.0250]))
        nodes.append(Node("rftip", 15, [0.0000, 0.0000, 0.0260]))
        nodes.append(Node("lfmetacarpal", 1, [-0.0170, 0.0000, 0.0440]))
        nodes.append(Node("lfknuckle", 17, [-0.0170, 0.0000, 0.0440]))
        nodes.append(Node("lfproximal", 18, [0.0000, 0.0000, 0.0000]))
        nodes.append(Node("lfmiddle", 19, [0.0000, 0.0000, 0.0450]))
        nodes.append(Node("lfdistal", 20, [0.0000, 0.0000, 0.0250]))
        nodes.append(Node("lftip", 21, [0.0000, 0.0000, 0.0260]))
        nodes.append(Node("thbase", 1, [0.0340, -0.0090, 0.0290]))
        nodes.append(Node("thproximal", 23, [0.0000, 0.0000, 0.0000]))
        nodes.append(Node("thhub", 24, [0.0000, 0.0000, 0.0380]))
        nodes.append(Node("thmiddle", 25, [0.0000, 0.0000, 0.0000]))
        nodes.append(Node("thdistal", 26, [0.0000, 0.0000, 0.0320]))
        nodes.append(Node("thtip", 27, [0.0000, 0.0000, 0.0275]))

        joints.append(Joint("WRJ1", 0, [0.0000, 1.0000, 0.0000], -0.4890, 0.1400))
        joints.append(Joint("WRJ0", 1, [1.0000, 0.0000, 0.0000], -0.6890, 0.4890))
        joints.append(Joint("FFJ3", 2, [0.0000, 1.0000, 0.0000], -0.3490, 0.3490))
        joints.append(Joint("FFJ2", 3, [1.0000, 0.0000, 0.0000], 0.0000, 1.5710))
        joints.append(Joint("FFJ1", 4, [1.0000, 0.0000, 0.0000], 0.0000, 1.5710))
        joints.append(Joint("FFJ0", 5, [1.0000, 0.0000, 0.0000], 0.0000, 1.5710))
        joints.append(Joint("MFJ3", 7, [0.0000, 1.0000, 0.0000], -0.3490, 0.3490))
        joints.append(Joint("MFJ2", 8, [1.0000, 0.0000, 0.0000], 0.0000, 1.5710))
        joints.append(Joint("MFJ1", 9, [1.0000, 0.0000, 0.0000], 0.0000, 1.5710))
        joints.append(Joint("MFJ0", 10, [1.0000, 0.0000, 0.0000], 0.0000, 1.5710))
        joints.append(Joint("RFJ3", 12, [0.0000, 1.0000, 0.0000], -0.3490, 0.3490))
        joints.append(Joint("RFJ2", 13, [1.0000, 0.0000, 0.0000], 0.0000, 1.5710))
        joints.append(Joint("RFJ1", 14, [1.0000, 0.0000, 0.0000], 0.0000, 1.5710))
        joints.append(Joint("RFJ0", 15, [1.0000, 0.0000, 0.0000], 0.0000, 1.5710))
        joints.append(Joint("LFJ4", 17, [0.5710, 0.0000, 0.8210], 0.0000, 0.7850))
        joints.append(Joint("LFJ3", 18, [0.0000, 1.0000, 0.0000], -0.3490, 0.3490))
        joints.append(Joint("LFJ2", 19, [1.0000, 0.0000, 0.0000], 0.0000, 1.5710))
        joints.append(Joint("LFJ1", 20, [1.0000, 0.0000, 0.0000], 0.0000, 1.5710))
        joints.append(Joint("LFJ0", 21, [1.0000, 0.0000, 0.0000], 0.0000, 1.5710))
        joints.append(Joint("THJ4", 23, [0.0000, 0.0000, -1.0000], -1.0470, 1.0470))
        joints.append(Joint("THJ3", 24, [1.0000, 0.0000, 0.0000], 0.0000, 1.2220))
        joints.append(Joint("THJ2", 25, [1.0000, 0.0000, 0.0000], -0.2090, 0.2090))
        joints.append(Joint("THJ1", 26, [0.0000, 1.0000, 0.0000], -0.5240, 0.5240))
        joints.append(Joint("THJ0", 27, [0.0000, 1.0000, 0.0000], -1.5710, 0.0000))

        self.nodes = nodes
        self.joints = joints

        self.dof: int = len(joints)

        axes = torch.stack([torch.tensor(j.axis) for j in joints], dim=0)
        self.axes = nn.Parameter(axes, requires_grad=False)

        min_rad = torch.tensor([j.min_rad for j in joints])
        max_rad = torch.tensor([j.max_rad for j in joints])
        self.min_rad = nn.Parameter(min_rad, requires_grad=False)
        self.max_rad = nn.Parameter(max_rad, requires_grad=False)

        translation = torch.stack([torch.tensor(n.translation) for n in nodes], dim=0)
        self.translation = nn.Parameter(translation, requires_grad=False)

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        """Forward kinematics.

        Args:
            angles (torch.Tensor): 24 joint angles.

        Returns:
            torch.Tensor: 3D coordinates of 21 keypoints.
        """
        angles = torch.clamp(angles, self.min_rad, self.max_rad)
        rotation = rotation_matrix_from_angle_axis(angles, self.axes)
        translation = self.translation

        rotation_wrist = rotation[..., 0, :, :]
        translation_wrist = translation[..., 0, :]

        rotation_palm, translation_palm = multiply_transform(
            (rotation_wrist, translation_wrist),
            (rotation[..., 1, :, :], translation[..., 1, :]),
        )

        rotation_ffknuckle, translation_ffknuckle = multiply_transform(
            (rotation_palm, translation_palm),
            (rotation[..., 2, :, :], translation[..., 2, :]),
        )
        rotation_ffproximal, translation_ffproximal = multiply_transform(
            (rotation_ffknuckle, translation_ffknuckle),
            (rotation[..., 3, :, :], translation[..., 3, :]),
        )
        rotation_ffmiddle, translation_ffmiddle = multiply_transform(
            (rotation_ffproximal, translation_ffproximal),
            (rotation[..., 4, :, :], translation[..., 4, :]),
        )
        rotation_ffdistal, translation_ffdistal = multiply_transform(
            (rotation_ffmiddle, translation_ffmiddle),
            (rotation[..., 5, :, :], translation[..., 5, :]),
        )
        translation_fftip = translation_ffdistal + torch.mv(
            rotation_ffdistal, translation[..., 6, :]
        )

        rotation_mfknuckle, translation_mfknuckle = multiply_transform(
            (rotation_palm, translation_palm),
            (rotation[..., 6, :, :], translation[..., 7, :]),
        )
        rotation_mfproximal, translation_mfproximal = multiply_transform(
            (rotation_mfknuckle, translation_mfknuckle),
            (rotation[..., 7, :, :], translation[..., 8, :]),
        )
        rotation_mfmiddle, translation_mfmiddle = multiply_transform(
            (rotation_mfproximal, translation_mfproximal),
            (rotation[..., 8, :, :], translation[..., 9, :]),
        )
        rotation_mfdistal, translation_mfdistal = multiply_transform(
            (rotation_mfmiddle, translation_mfmiddle),
            (rotation[..., 9, :, :], translation[..., 10, :]),
        )
        translation_mftip = translation_mfdistal + torch.mv(
            rotation_mfdistal, translation[..., 11, :]
        )

        rotation_rfknuckle, translation_rfknuckle = multiply_transform(
            (rotation_palm, translation_palm),
            (rotation[..., 10, :, :], translation[..., 12, :]),
        )
        rotation_rfproximal, translation_rfproximal = multiply_transform(
            (rotation_rfknuckle, translation_rfknuckle),
            (rotation[..., 11, :, :], translation[..., 13, :]),
        )
        rotation_rfmiddle, translation_rfmiddle = multiply_transform(
            (rotation_rfproximal, translation_rfproximal),
            (rotation[..., 12, :, :], translation[..., 14, :]),
        )
        rotation_rfdistal, translation_rfdistal = multiply_transform(
            (rotation_rfmiddle, translation_rfmiddle),
            (rotation[..., 13, :, :], translation[..., 15, :]),
        )
        translation_rftip = translation_rfdistal + torch.mv(
            rotation_rfdistal, translation[..., 16, :]
        )

        rotation_lfmetacarpal, translation_lfmetacarpal = multiply_transform(
            (rotation_palm, translation_palm),
            (rotation[..., 14, :, :], translation[..., 17, :]),
        )
        rotation_lfknuckle, translation_lfknuckle = multiply_transform(
            (rotation_lfmetacarpal, translation_lfmetacarpal),
            (rotation[..., 15, :, :], translation[..., 18, :]),
        )
        rotation_lfproximal, translation_lfproximal = multiply_transform(
            (rotation_lfknuckle, translation_lfknuckle),
            (rotation[..., 16, :, :], translation[..., 19, :]),
        )
        rotation_lfmiddle, translation_lfmiddle = multiply_transform(
            (rotation_lfproximal, translation_lfproximal),
            (rotation[..., 17, :, :], translation[..., 20, :]),
        )
        rotation_lfdistal, translation_lfdistal = multiply_transform(
            (rotation_lfmiddle, translation_lfmiddle),
            (rotation[..., 18, :, :], translation[..., 21, :]),
        )
        translation_lftip = translation_lfdistal + torch.mv(
            rotation_lfdistal, translation[..., 22, :]
        )

        rotation_thbase, translation_thbase = multiply_transform(
            (rotation_palm, translation_palm),
            (rotation[..., 19, :, :], translation[..., 23, :]),
        )
        rotation_thproximal, translation_thproximal = multiply_transform(
            (rotation_thbase, translation_thbase),
            (rotation[..., 20, :, :], translation[..., 24, :]),
        )
        rotation_thhub, translation_thhub = multiply_transform(
            (rotation_thproximal, translation_thproximal),
            (rotation[..., 21, :, :], translation[..., 25, :]),
        )
        rotation_thmiddle, translation_thmiddle = multiply_transform(
            (rotation_thhub, translation_thhub),
            (rotation[..., 22, :, :], translation[..., 26, :]),
        )
        rotation_thdistal, translation_thdistal = multiply_transform(
            (rotation_thmiddle, translation_thmiddle),
            (rotation[..., 23, :, :], translation[..., 27, :]),
        )
        translation_thtip = translation_thdistal + torch.mv(
            rotation_thdistal, translation[..., 28, :]
        )

        return (
            torch.stack(
                [
                    translation_palm,
                    translation_thbase,
                    translation_thmiddle,
                    translation_thdistal,
                    translation_thtip,
                    translation_ffknuckle,
                    translation_ffmiddle,
                    translation_ffdistal,
                    translation_fftip,
                    translation_mfknuckle,
                    translation_mfmiddle,
                    translation_mfdistal,
                    translation_mftip,
                    translation_rfknuckle,
                    translation_rfmiddle,
                    translation_rfdistal,
                    translation_rftip,
                    translation_lfknuckle,
                    translation_lfmiddle,
                    translation_lfdistal,
                    translation_lftip,
                ],
                dim=-2,
            )
            - translation_palm.detach().clone()
        )

    def zero_pose(self) -> torch.Tensor:
        angles = torch.zeros(self.dof)
        return self.forward(angles)

    def index_of_joint(self, name: str) -> int:
        joint_names = [j.name for j in self.joints]
        return joint_names.index(name)


if __name__ == "__main__":
    # for i in range(10):
    #     shadow = ShadowHandModule()
    #     result = shadow.forward(torch.ones(23) * 0.05)
    #     print(i)
    #     print(result)

    shadow = ShadowHandModule()
    print(shadow.zero_pose())
