import shutil
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
from workspace import MultiViewSystem


class MultiCameraManager(object):
    serial_numbers: List[str]

    def __init__(
        self,
        save_dir: Path = Path("."),
        workspace: MultiViewSystem = None,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        enable_color_stream: bool = True,
        enable_depth_stream: bool = True,
        enable_hardware_sync: bool = True,
        enable_record_to_file: bool = False,
        align_to: str = "color",
    ) -> None:
        assert align_to == "color"
        self.save_dir = save_dir
        self.workspace = MultiViewSystem(save_dir)
        self.width = width
        self.height = height
        self.fps = fps

        self.enable_color_stream = enable_color_stream
        self.enable_depth_stream = enable_depth_stream
        self.enable_hardware_sync = enable_hardware_sync
        self.enable_record_to_file = enable_record_to_file

        self.align = rs.align(rs.stream.color)

        self.pipelines = []
        self.configs = []
        self.profiles = []

        ctx = rs.context()
        devices = ctx.query_devices()
        self.n_device = len(devices)
        self.devices = devices

        print(f"Number of devices found: {self.n_device}")

        serial_numbers = [d.get_info(rs.camera_info.serial_number) for d in devices]
        self.serial_numbers = serial_numbers

        print("Serial numbers:")
        for i, number in enumerate(serial_numbers):
            print(f"[{i}] - {number}")

        if self.enable_hardware_sync:
            self.hardware_sync()

        for number in serial_numbers:
            pipeline = rs.pipeline()
            config = self.create_config(number)

            self.pipelines.append(pipeline)
            self.configs.append(config)

            self.workspace.add_camera(number)

        self.workspace.reset()

    def hardware_sync(self):
        if self.n_device == 1:
            return

        master, slave = 1, 2
        for i, device in enumerate(self.devices):
            cam_sync_mode = master if i == 0 else slave
            sensor = device.first_depth_sensor()
            sensor.set_option(rs.option.inter_cam_sync_mode, cam_sync_mode)

            sensor = device.first_color_sensor()
            sensor.set_option(rs.option.auto_exposure_priority, 0)

    def create_config(self, serial_number: str):
        config = rs.config()
        config.enable_device(serial_number)
        if self.enable_color_stream:
            config.enable_stream(
                rs.stream.color,
                self.width,
                self.height,
                rs.format.bgr8,
                self.fps,
            )
        if self.enable_depth_stream:
            config.enable_stream(
                rs.stream.depth,
                self.width,
                self.height,
                rs.format.z16,
                self.fps,
            )
        if self.enable_record_to_file:
            config.enable_record_to_file(
                str(self.workspace.get_camera_bag_path(serial_number))
            )

        return config

    def start(self) -> None:
        for number, pipeline, config in zip(
            self.serial_numbers, self.pipelines, self.configs
        ):
            print(config)
            profile = pipeline.start(config)
            self.profiles.append(profile)

            color_profile = profile.get_stream(rs.stream.color)
            intrinsic = color_profile.as_video_stream_profile().get_intrinsics()
            print(intrinsic)

            self.workspace.set_camera_parameter(
                number,
                width=intrinsic.width,
                height=intrinsic.height,
                fx=intrinsic.fx,
                fy=intrinsic.fy,
                cx=intrinsic.ppx,
                cy=intrinsic.ppy,
            )
            time.sleep(3)

    def wait_for_frames(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        depth_frames: List[rs.frame] = []
        color_frames: List[rs.frame] = []
        timestamps = []

        for i, pipeline in enumerate(self.pipelines):
            frames = pipeline.poll_for_frames()
            frames = pipeline.wait_for_frames()
            frames = self.align.process(frames)

            timestamp = self.get_frame_timestamp(frames)
            timestamps.append(timestamp)
            
            print(i, timestamp)

            # depth_frames.append(frames.get_depth_frame())
            # color_frames.append(frames.get_color_frame())
            depth_frame = self.frame_to_array(frames.get_depth_frame())
            color_frame = self.frame_to_array(frames.get_color_frame()) 
            depth_frames.append(depth_frame)
            color_frames.append(color_frame)
            
            cv2.imshow(f"frame_{i}", color_frame)
        
        k = cv2.waitKey(1) & 0xFF
        if k == ord('c'):
            timestamp = timestamps[0]
            for i, frame in enumerate(color_frames):
                # if not frame.is_frame():
                #     continue
                # frame = self.frame_to_array(frame)
                cv2.imwrite(
                    str(
                        self.workspace.get_camera_color_dir(self.serial_numbers[i])
                        / f"{timestamp}.png"
                    ),
                    frame,
                )
            for i, frame in enumerate(depth_frames):
                # if not frame.is_frame():
                #     continue
                # frame = self.frame_to_array(frame)
                cv2.imwrite(
                    str(
                        self.workspace.get_camera_depth_dir(self.serial_numbers[i])
                        / f"{timestamp}.png"
                    ),
                    frame,
                )

        return depth_frames, color_frames

    @classmethod
    def get_frame_timestamp(cls, frame) -> int:
        return frame.get_frame_metadata(rs.frame_metadata_value.backend_timestamp)

    @classmethod
    def frame_to_array(cls, frame: rs.frame) -> Optional[np.ndarray]:
        return np.asanyarray(frame.get_data()) if frame.is_frame() else None
