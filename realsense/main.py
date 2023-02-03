import argparse
from pathlib import Path
from typing import Optional

from tqdm import tqdm
import time

from manager import MultiCameraManager


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="./workspace")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--disable_color_stream", action="store_true")
    parser.add_argument("--disable_depth_stream", action="store_true")
    parser.add_argument("--enable_record_to_file", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    manager = MultiCameraManager(
        save_dir=Path(args.save_dir),
        width=args.width,
        height=args.height,
        fps=args.fps,
        enable_color_stream=not args.disable_color_stream,
        enable_depth_stream=not args.disable_depth_stream,
        enable_record_to_file=args.enable_record_to_file,
    )
    manager.start()

    if not args.enable_record_to_file:
        for i in tqdm(range(1000)):
            frames = manager.wait_for_frames()

    else:
        input()
