from workspace import MultiViewSystem
from pathlib import Path
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=str, default="./workspace")
    parser.add_argument("--calibration", type=str, default="./calibration")
    parser.add_argument("--board", type=str, default="./charuco.yaml")
    args = parser.parse_args()

    board_config_path = Path(args.board)
    multiview = MultiViewSystem.from_workspace(Path(args.workspace))
    multiview.create_calibration_workspace(
        Path(args.calibration), force=True, board_config_path=board_config_path
    )
