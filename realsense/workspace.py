import json
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import open3d as o3d


class MultiViewSystem:
    names: List[str]
    cameras: List[o3d.camera.PinholeCameraParameters]
    parameters: Dict[str, o3d.camera.PinholeCameraParameters]
    workspace_path: Path

    def __init__(
        self,
        workspace_path: Path = Path("."),
        cameras: Optional[List[o3d.camera.PinholeCameraParameters]] = None,
        names: Optional[List[str]] = None,
    ) -> None:
        self.workspace_path = workspace_path
        if cameras is None:
            self.cameras: List[o3d.camera.PinholeCameraParameters] = []
            self.names: List[str] = []
        else:
            assert names is None or len(cameras) == len(names)
            self.cameras = cameras
            if names is None:
                self.names = [f"cam{i}" for i in range(len(cameras))]
            else:
                self.names = names

        self.parameters = {}

    @property
    def n_cameras(self) -> int:
        return len(self.cameras)

    @property
    def named_cameras(self) -> List[Tuple[str, o3d.camera.PinholeCameraParameters]]:
        yield from zip(self.names, self.cameras)

    @classmethod
    def from_workspace(cls, workspace_path: Path) -> "MultiViewSystem":
        assert workspace_path.is_dir(), "Invalid workspace path"
        cameras = []
        names = []
        for camera_path in workspace_path.glob("cameras/*.json"):
            camera = o3d.io.read_pinhole_camera_parameters(str(camera_path))
            cameras.append(camera)
            names.append(camera_path.stem)
        return cls(workspace_path, cameras, names)

    @classmethod
    def from_multical_result(cls, calibration_path: Path) -> "MultiViewSystem":
        # assert it's json file
        assert calibration_path.is_file(), "Invalid calibration path"
        # read json file
        with open(calibration_path, "r") as f:
            calibration = json.load(f)
        raise NotImplementedError("Not implemented yet")

    def save(self, workspace_path: Path) -> None:
        assert workspace_path.is_dir(), "Invalid workspace path"
        for name, camera in self.named_cameras:
            o3d.io.write_pinhole_camera_parameters(
                str(workspace_path / "cameras" / f"{name}.json"), camera
            )

    def add_camera(
        self, name: str, camera: Optional[o3d.camera.PinholeCameraParameters] = None
    ) -> None:
        if camera is None:
            camera = o3d.camera.PinholeCameraParameters()
        self.names.append(name)
        self.cameras.append(camera)

    def get_camera(self, index: int) -> o3d.camera.PinholeCameraParameters:
        assert 0 <= index < self.n_cameras, "Invalid camera index"
        return self.cameras[index]

    def set_camera_parameter(
        self,
        name: str,
        parameter: Optional[o3d.camera.PinholeCameraParameters] = None,
        intrinsic: Optional[o3d.camera.PinholeCameraIntrinsic] = None,
        extrinsic: Optional[np.ndarray] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fx: Optional[float] = None,
        fy: Optional[float] = None,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
    ) -> None:
        if parameter is not None:
            self.parameters[name] = parameter
            return

        parameter = o3d.camera.PinholeCameraParameters()

        if intrinsic is None:
            assert width is not None and height is not None
            assert fx is not None and fy is not None
            assert cx is not None and cy is not None
            intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
        parameter.intrinsic = intrinsic

        if extrinsic is None:
            extrinsic = np.eye(4)
        parameter.extrinsic = extrinsic

        self.parameters[name] = parameter

    def set_camera_intrinsic(
        self, name: str, intrinsic: o3d.camera.PinholeCameraIntrinsic
    ) -> None:
        self.parameters[name].intrinsic = intrinsic

    def set_camera_extrinsic(self, name: str, extrinsic: np.ndarray) -> None:
        self.parameters[name].extrinsic = extrinsic

    def get_camera_name(self, index: int) -> str:
        assert 0 <= index < self.n_cameras, "Invalid camera index"
        return self.names[index]

    def get_camera_intrinsic(self, index: int) -> o3d.camera.PinholeCameraIntrinsic:
        return self.get_camera(index).intrinsic

    def get_camera_extrinsic(self, index: int) -> np.ndarray:
        return self.get_camera(index).extrinsic

    def get_parameter_dir(self) -> Path:
        return self.workspace_path / "cameras"

    def get_rosbag_dir(self) -> Path:
        return self.workspace_path / "rosbags"

    def get_camera_config_path(self, name: str) -> Path:
        return self.workspace_path / "cameras" / f"{name}.json"

    def get_camera_depth_dir(self, name: str) -> Path:
        return self.workspace_path / "data" / name / "depth"

    def get_camera_color_dir(self, name: str) -> Path:
        return self.workspace_path / "data" / name / "color"

    def get_camera_bag_path(self, name: str) -> Path:
        return self.workspace_path / "rosbags" / f"{name}.bag"

    def create_workspace(self) -> None:
        self.workspace_path.mkdir(parents=True, exist_ok=True)
        self.get_parameter_dir().mkdir(parents=True, exist_ok=True)
        self.get_rosbag_dir().mkdir(parents=True, exist_ok=True)
        for name in self.names:
            self.get_camera_depth_dir(name).mkdir(parents=True, exist_ok=True)
            self.get_camera_color_dir(name).mkdir(parents=True, exist_ok=True)

    def reset(self) -> None:
        for name in self.names:
            shutil.rmtree(self.get_camera_depth_dir(name), ignore_errors=True)
            shutil.rmtree(self.get_camera_color_dir(name), ignore_errors=True)
        self.get_rosbag_dir().mkdir(parents=True, exist_ok=True)
        for name in self.names:
            self.get_camera_depth_dir(name).mkdir(parents=True, exist_ok=True)
            self.get_camera_color_dir(name).mkdir(parents=True, exist_ok=True)
