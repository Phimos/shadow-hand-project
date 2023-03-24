import argparse
from pathlib import Path
import json
from workspace import MultiViewSystem


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=str, default="./workspace")
    parser.add_argument(
        "--intrinsic-file", type=str, default="./calibration/intrinsic.json"
    )
    args = parser.parse_args()

    multiview = MultiViewSystem.from_workspace(Path(args.workspace))

    with open(Path(args.intrinsic_file), "r") as f:
        intrinsic = json.load(f)

    for name in intrinsic["cameras"].keys():
        intrinsic_matrix = multiview.get_camera_intrinsic_by_name(name).intrinsic_matrix
        intrinsic["cameras"][name]["K"] = intrinsic_matrix.tolist()
        intrinsic["cameras"][name]["dist"] = [0.0, 0.0, 0.0, 0.0, 0.0]

    with open(Path(args.intrinsic_file), "w") as f:
        json.dump(intrinsic, f, indent=4)
