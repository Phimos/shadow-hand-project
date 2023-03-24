import json
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union

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
        for name, camera in self.named_cameras:
            self.parameters[name] = camera

    @property
    def n_cameras(self) -> int:
        return len(self.cameras)

    @property
    def named_cameras(self) -> List[Tuple[str, o3d.camera.PinholeCameraParameters]]:
        yield from zip(self.names, self.cameras)

    def sort_camera_by_name(self):
        self.names, self.cameras = zip(*sorted(zip(self.names, self.cameras)))

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

    def add_camera(
        self, name: str, camera: Optional[o3d.camera.PinholeCameraParameters] = None
    ) -> None:
        if camera is None:
            camera = o3d.camera.PinholeCameraParameters()
        self.names.append(name)
        self.cameras.append(camera)

    def get_camera(self, name: Union[int, str]) -> o3d.camera.PinholeCameraParameters:
        if isinstance(name, str):
            return self.cameras[self.names.index(name)]
        elif isinstance(name, int):    
            assert 0 <= name < self.n_cameras, "Invalid camera index"
            return self.cameras[name]
        else:
            raise TypeError("Invalid camera name type, expected str or int")

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
        self.cameras[self.names.index(name)] = parameter

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

    def get_camera_intrinsic_by_name(
        self, name: str
    ) -> o3d.camera.PinholeCameraIntrinsic:
        return self.cameras[self.names.index(name)].intrinsic

    def get_camera_extrinsic_by_name(self, name: str) -> np.ndarray:
        return self.cameras[self.names.index(name)].extrinsic

    def get_camera_config_path(self, name: str) -> Path:
        return self.workspace_path / "cameras" / f"{name}.json"

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

    def get_camera_config_dir(self) -> Path:
        return self.workspace_path / "cameras"

    def get_camera_bag_path(self, name: str) -> Path:
        return self.workspace_path / "rosbags" / f"{name}.bag"

    def create_workspace(self) -> None:
        self.workspace_path.mkdir(parents=True, exist_ok=True)
        self.get_parameter_dir().mkdir(parents=True, exist_ok=True)
        self.get_rosbag_dir().mkdir(parents=True, exist_ok=True)
        for name in self.names:
            self.get_camera_depth_dir(name).mkdir(parents=True, exist_ok=True)
            self.get_camera_color_dir(name).mkdir(parents=True, exist_ok=True)

    def create_calibration_workspace(
        self,
        workspace_path: Path,
        force: bool = True,
        board_config_path: Optional[Path] = None,
    ) -> None:
        if force:
            shutil.rmtree(workspace_path, ignore_errors=True)
        else:
            assert not workspace_path.exists(), "Workspace already exists"
        workspace_path.mkdir(parents=True, exist_ok=True)

        for name in self.names:
            shutil.copytree(self.get_camera_color_dir(name), workspace_path / name)

        if board_config_path is not None:
            assert board_config_path.exists(), "Given board config path does not exist"
            assert board_config_path.is_file(), "Given board config path is not a file"
            assert (
                board_config_path.suffix == ".yaml"
            ), "Given board config path is not a yaml file"

            shutil.copy(board_config_path, workspace_path / "boards.yaml")

    def dump_camera_parameters(self) -> None:
        for name, camera in self.named_cameras:
            o3d.io.write_pinhole_camera_parameters(
                str(self.get_camera_config_path(name)), camera
            )

    def load_calibration(self, calibration_path: Path) -> None:
        assert calibration_path.exists(), "Given calibration path does not exist"
        assert calibration_path.is_file(), "Given calibration path is not a file"
        assert (
            calibration_path.suffix == ".json"
        ), "Given calibration path is not a json file"
        with open(calibration_path, "r") as file:
            data = json.load(file)

        for name, info in data["cameras"].items():
            width = info["image_size"][0]
            height = info["image_size"][1]
            fx = info["K"][0][0]
            fy = info["K"][1][1]
            cx = info["K"][0][2]
            cy = info["K"][1][2]
            intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
            self.set_camera_intrinsic(name, intrinsic)

        extrinsics: Dict[str, np.ndarray] = {}
        for name, info in data["camera_poses"].items():
            if name.count("_to_") != 0:
                continue
            extrinsics[name] = np.eye(4)
            extrinsics[name][:3, :3] = np.array(info["R"])
            extrinsics[name][:3, 3] = np.array(info["T"])

        def all_extrinsics_know() -> bool:
            for name in self.names:
                if name not in extrinsics:
                    return False
            return True

        while not all_extrinsics_know():
            for name, info in data["camera_poses"].items():
                if name.count("_to_") == 1:
                    source = name.split("_to_")[0]
                    target = name.split("_to_")[1]

                    pose = np.eye(4)
                    pose[:3, :3] = np.array(info["R"])
                    pose[:3, 3] = np.array(info["T"])

                    if source in extrinsics and target not in extrinsics:
                        extrinsics[target] = np.dot(
                            extrinsics[source], np.linalg.inv(pose)
                        )
                    elif source not in extrinsics and target in extrinsics:
                        print(pose, extrinsics[target])
                        extrinsics[source] = np.dot(extrinsics[target], pose)
                    else:
                        continue

        for name, extrinsic in extrinsics.items():
            self.set_camera_extrinsic(name, extrinsic)

    def valid_calibration(self) -> bool:
        return not all(
            [np.array_equal(camera.extrinsic, np.eye(4)) for camera in self.cameras]
        )

    def reset(self) -> None:
        for name in self.names:
            shutil.rmtree(self.get_camera_depth_dir(name), ignore_errors=True)
            shutil.rmtree(self.get_camera_color_dir(name), ignore_errors=True)
        shutil.rmtree(self.get_rosbag_dir(), ignore_errors=True)
        shutil.rmtree(self.get_camera_config_dir(), ignore_errors=True)

        for name in self.names:
            self.get_camera_depth_dir(name).mkdir(parents=True, exist_ok=True)
            self.get_camera_color_dir(name).mkdir(parents=True, exist_ok=True)
        self.get_rosbag_dir().mkdir(parents=True, exist_ok=True)
        self.get_camera_config_dir().mkdir(parents=True, exist_ok=True)
