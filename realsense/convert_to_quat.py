import scipy.spatial.transform.rotation as R
from workspace import MultiViewSystem
import numpy as np
from pathlib import Path

if __name__ == "__main__":
    LINK_TO_OPTICAL_FRAME_TRANSLATION = np.array([0.0, 0.015, 0.0])
    LINK_TO_OPTICAL_FRAME_ROTATION = R.Rotation.from_quat(
        [0.5, -0.5, 0.5, -0.5]
    ).as_matrix()

    OPTICAL_FRAME_TO_LINK_TRANSLATION = np.array([0.015, 0.0, 0.0])
    OPTICAL_FRAME_TO_LINK_ROTATION = R.Rotation.from_quat(
        [0.5, -0.5, 0.5, 0.5]
    ).as_matrix()

    multiview = MultiViewSystem.from_workspace(Path("./workspace"))
    multiview.sort_camera_by_name()

    meshs = []
    root = multiview.names[0]

    for name, camera in multiview.named_cameras:
        if name == root:
            continue

        print(f"Camera {name}:")
        extrinsic = camera.extrinsic
        translation = np.linalg.inv(extrinsic)[:3, 3]
        rotation = np.linalg.inv(extrinsic)[:3, :3]

        translation = (
            LINK_TO_OPTICAL_FRAME_ROTATION @ translation
            + LINK_TO_OPTICAL_FRAME_TRANSLATION
        )
        rotation = LINK_TO_OPTICAL_FRAME_ROTATION @ rotation

        translation = rotation @ OPTICAL_FRAME_TO_LINK_TRANSLATION + translation
        rotation = rotation @ OPTICAL_FRAME_TO_LINK_ROTATION

        print(f"Translation: {translation}")
        print(f"Rotation(quat): {R.Rotation.from_matrix(rotation).as_quat()})")
