import scipy.spatial.transform.rotation as R
import json
import open3d as o3d
import numpy as np

if __name__ == "__main__":
    LINK_TO_OPTICAL_FRAME_TRANSLATION = np.array([0.0, 0.015, 0.0])
    LINK_TO_OPTICAL_FRAME_ROTATION = R.Rotation.from_quat(
        [0.5, -0.5, 0.5, -0.5]
    ).as_matrix()

    OPTICAL_FRAME_TO_LINK_TRANSLATION = np.array([0.015, 0.0, 0.0])
    OPTICAL_FRAME_TO_LINK_ROTATION = R.Rotation.from_quat(
        [0.5, -0.5, 0.5, 0.5]
    ).as_matrix()

    with open("calibration.json") as f:
        calibration = json.load(f)

    intrinsics = {}
    extrinsics = {}

    for name, info in calibration["cameras"].items():
        width = info["image_size"][0]
        height = info["image_size"][1]
        fx = info["K"][0][0]
        fy = info["K"][1][1]
        cx = info["K"][0][2]
        cy = info["K"][1][2]
        intrinsics[name] = o3d.camera.PinholeCameraIntrinsic(
            width, height, fx, fy, cx, cy
        )
        o3d.io.write_pinhole_camera_intrinsic(f"{name}.json", intrinsics[name])

    for name, info in calibration["camera_poses"].items():
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = np.array(info["R"])
        extrinsic[:3, 3] = np.array(info["T"])
        extrinsics[name] = extrinsic

    cameras = {}
    cam0 = list(calibration["camera_poses"].keys())[0]
    for name, intrinsic in intrinsics.items():
        cameras[name] = o3d.camera.PinholeCameraParameters()
        cameras[name].intrinsic = intrinsic
        if name == cam0:
            cameras[name].extrinsic = extrinsics[name]
        else:
            cameras[name].extrinsic = extrinsics[f"{name}_to_{cam0}"]

    meshs = []

    for name, camera in cameras.items():
        print(f"Camera {name}:")
        print(f"Translation: {camera.extrinsic[:3, 3]}")
        print(
            f"Rotation(quat): {R.Rotation.from_matrix(camera.extrinsic[:3, :3]).as_quat()})"
        )

        translation = camera.extrinsic[:3, 3]
        rotation = camera.extrinsic[:3, :3]

        translation = (
            LINK_TO_OPTICAL_FRAME_ROTATION @ translation
            + LINK_TO_OPTICAL_FRAME_TRANSLATION
        )
        rotation = LINK_TO_OPTICAL_FRAME_ROTATION @ rotation

        translation = rotation @ OPTICAL_FRAME_TO_LINK_TRANSLATION + translation
        rotation = rotation @ OPTICAL_FRAME_TO_LINK_ROTATION

        print(translation, rotation)

        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=[0, 0, 0]
        )
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = translation

        mesh.transform(transform)
        meshs.append(mesh)

        # translation = translation
        # rotation = rotation @ AXIS_ROTATION

        print(f"Translation: {translation}")
        print(f"Rotation(quat): {R.Rotation.from_matrix(rotation).as_quat()})")
        print("")

    o3d.visualization.draw_geometries(meshs)
    # visualize name, camera
