from workspace import MultiViewSystem
import argparse
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=str, default="./workspace")
    parser.add_argument(
        "--calibration-json", type=str, default="./calibration/calibration.json"
    )
    args = parser.parse_args()

    multiview = MultiViewSystem.from_workspace(Path(args.workspace))
    multiview.load_calibration(Path(args.calibration_json))
    multiview.dump_camera_parameters()
