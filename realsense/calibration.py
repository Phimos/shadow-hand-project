import argparse
import os
import pathlib
import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pyrealsense2 as rs
from tqdm import tqdm
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="./data")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--disable_color_stream", action="store_true")
    parser.add_argument("--disable_depth_stream", action="store_true")
    parser.add_argument("--enable_record_to_file", action="store_true")
    return parser.parse_args()


class MultiCameraManager(object):
    def __init__(
        self,
        save_dir: Optional[Path] = Path("./data"),
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        enable_color_stream: bool = True,
        enable_depth_stream: bool = True,
        record_to_file: bool = False,
    ) -> None:
        self.save_dir = save_dir
        self.width = width
        self.height = height
        self.fps = fps

        self.enable_color_stream = enable_color_stream
        self.enable_depth_stream = enable_depth_stream
        self.enable_record_to_file = record_to_file

        self.pipelines = []
        self.configs = []
        self.profiles = []

        ctx = rs.context()

        devices = ctx.query_devices()
        self.n_device = len(devices)

        print(f"Number of devices found: {self.n_device}")

        serial_numbers = [d.get_info(rs.camera_info.serial_number) for d in devices]
        self.serial_numbers = serial_numbers

        print("Serial numbers:")
        for i, number in enumerate(serial_numbers):
            print(f"[{i}] - {number}")

        for i, device in enumerate(devices):
            master_or_slave = 1 if i == 0 else 2
            sensor = device.first_depth_sensor()
            sensor.set_option(rs.option.inter_cam_sync_mode, master_or_slave)

        for number in serial_numbers:
            pipeline = rs.pipeline()
            config = self.create_config(number)

            self.pipelines.append(pipeline)
            self.configs.append(config)

        if self.save_dir is not None:
            shutil.rmtree(self.save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)

        for i in range(self.n_device):
            (self.save_dir / f"cam{i}" / "image_color").mkdir(
                parents=True, exist_ok=True
            )
            (self.save_dir / f"cam{i}" / "image_depth").mkdir(
                parents=True, exist_ok=True
            )

    def create_config(self, serial_number):
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
                str(
                    self.save_dir / f"cam{self.serial_numbers.index(serial_number)}.bag"
                )
            )

        return config

    def start(self) -> None:
        for pipeline, config in zip(self.pipelines, self.configs):
            print(config)
            profile = pipeline.start(config)
            self.profiles.append(profile)

            color_profile = profile.get_stream(rs.stream.color)
            intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
            print(intrinsics)

            time.sleep(3)

    def wait_for_frames(self):
        depth_frames, color_frames = [], []
        timestamps = []

        for i, pipeline in enumerate(self.pipelines):
            subdir = self.save_dir / f"cam{i}"
            frames = pipeline.wait_for_frames()
            # frames = pipeline.poll_for_frames()

            timestamp = self.get_frame_timestamp(frames)
            timestamps.append(timestamp)

            depth_frame = frames.get_depth_frame()
            depth_frame = self.frame_to_array(depth_frame) if depth_frame else None
            depth_frames.append(depth_frame)
            # cv2.imwrite(str(subdir / "image_depth" / f"{timestamp}.png"), depth_frame)

            color_frame = frames.get_color_frame()
            color_frame = self.frame_to_array(color_frame) if color_frame else None
            color_frames.append(color_frame)
            # cv2.imwrite(str(subdir / "image_color" / f"{timestamp}.png"), color_frame)

        timestamp = timestamps[0]
        for i, frame in enumerate(color_frames):
            if frame is None:
                continue
            cv2.imwrite(
                str(self.save_dir / f"cam{i}" / "image_color" / f"{timestamp}.png"),
                frame,
            )
        for i, frame in enumerate(depth_frames):
            if frame is None:
                continue
            cv2.imwrite(
                str(self.save_dir / f"cam{i}" / "image_depth" / f"{timestamp}.png"),
                frame,
            )

        return depth_frames, color_frames

    @classmethod
    def get_frame_timestamp(cls, frame) -> int:
        return frame.get_frame_metadata(rs.frame_metadata_value.backend_timestamp)

    @classmethod
    def frame_to_array(cls, frame) -> np.ndarray:
        return np.asanyarray(frame.get_data())


if __name__ == "__main__":
    args = parse_args()
    manager = MultiCameraManager(
        save_dir=Path(args.save_dir),
        width=args.width,
        height=args.height,
        fps=args.fps,
        enable_color_stream=not args.disable_color_stream,
        enable_depth_stream=not args.disable_depth_stream,
    )
    manager.start()

    for i in tqdm(range(1000)):
        frames = manager.wait_for_frames()
