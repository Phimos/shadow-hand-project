import json
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import open3d as o3d
from workspace import MultiViewSystem
import os


# Load the cameras intrinsics & extrinsics
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
    intrinsics[name] = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    o3d.io.write_pinhole_camera_intrinsic(f"{name}.json", intrinsics[name])


for name, info in calibration["camera_poses"].items():
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = np.array(info["R"])
    extrinsic[:3, 3] = np.array(info["T"])
    extrinsics[name] = extrinsic



cameras = {}
os.makedirs("cameras", exist_ok=True)
cam0 = list(calibration['camera_poses'].keys())[0]
for name, intrinsic in intrinsics.items():
    cameras[name] = o3d.camera.PinholeCameraParameters()
    cameras[name].intrinsic = intrinsic
    if name == cam0:
        cameras[name].extrinsic = extrinsics[name]
    else:
        cameras[name].extrinsic = extrinsics[f"{name}_to_{cam0}"]
    o3d.io.write_pinhole_camera_parameters(f"cameras/{name}.json", cameras[name])

multiview = MultiViewSystem.from_workspace(Path("."))

# dataset_dir = Path("cam-4-depth")
dataset_dir = Path("../workspace/data")


def create_point_cloud(
    camera: o3d.camera.PinholeCameraParameters,
    depth: np.ndarray,
    color: Optional[np.ndarray] = None,
) -> o3d.geometry.PointCloud:
    """Create a point cloud from a depth image and a color image (optional).

    Args:
        camera (o3d.camera.PinholeCameraParameters): Camera parameters.
        depth (np.ndarray): Depth image.
        color (Optional[np.ndarray], optional): Color image. Defaults to None.

    Returns:
        o3d.geometry.PointCloud: Point cloud.
    """
    intrinsic, extrinsic = camera.intrinsic, camera.extrinsic
    if color is None:
        depth = o3d.geometry.Image(depth)
        pcd = o3d.geometry.PointCloud.create_from_depth_image(depth, intrinsic, extrinsic)
    else:
        depth, color = o3d.geometry.Image(depth), o3d.geometry.Image(color)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic, extrinsic)
    return pcd


pcds = []
# for name, intrinsic in intrinsics.items():
for name, camera in multiview.named_cameras:
    filepath_depth = dataset_dir / name / "depth" / "1673601622338.png"
    filepath_color = dataset_dir / name / "color" / "1673601622338.png"
    # filepath_depth = dataset_dir / name / "image_depth" / "1672997458773.png"
    # filepath_color = dataset_dir / name / "image_color" / "1672997458773.png"

    depth = o3d.io.read_image(str(filepath_depth))
    color = o3d.io.read_image(str(filepath_color))

    pcd = create_point_cloud(camera, depth, color)

    pcds.append(pcd)


class PointCloudMatcher(object):
    """Point cloud matcher using ICP."""

    def __init__(
        self,
        voxel_size: float = 0.01,
        max_correspondence_distance_coarse_scale: float = 15.0,
        max_correspondence_distance_fine_scale: float = 1.5,
    ) -> None:
        self.voxel_size = voxel_size
        self.max_correspondence_distance_coarse = max_correspondence_distance_coarse_scale * voxel_size
        self.max_correspondence_distance_fine = max_correspondence_distance_fine_scale * voxel_size

    def pairwise_registration(
        self, source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud
    ) -> Tuple[np.ndarray, np.ndarray]:
        icp_coarse = o3d.pipelines.registration.registration_icp(
            source,
            target,
            self.max_correspondence_distance_coarse,
            np.identity(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        )
        icp_fine = o3d.pipelines.registration.registration_icp(
            source,
            target,
            self.max_correspondence_distance_fine,
            icp_coarse.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        )
        transformation_icp = icp_fine.transformation
        information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            source,
            target,
            self.max_correspondence_distance_fine,
            icp_fine.transformation,
        )
        return transformation_icp, information_icp

    def full_registration(self, pcds: List[o3d.geometry.PointCloud]) -> o3d.pipelines.registration.PoseGraph:
        pose_graph = o3d.pipelines.registration.PoseGraph()
        odometry = np.identity(4)
        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
        n_pcds = len(pcds)
        for source_id in range(n_pcds):
            for target_id in range(source_id + 1, n_pcds):
                transformation_icp, information_icp = self.pairwise_registration(pcds[source_id], pcds[target_id])
                print("Build o3d.pipelines.registration.PoseGraph")
                if target_id == source_id + 1:  # odometry case
                    odometry = np.dot(transformation_icp, odometry)
                    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry)))
                    uncertain = False
                else:  # loop closure case
                    uncertain = True
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(
                        source_id,
                        target_id,
                        transformation_icp,
                        information_icp,
                        uncertain=uncertain,
                    )
                )
        return pose_graph

    def fit(self, pcds: List[o3d.geometry.PointCloud]) -> None:
        pcds = deepcopy(pcds)
        for i, pcd in enumerate(pcds):
            pcd = pcd.voxel_down_sample(self.voxel_size)
            pcd.estimate_normals()
            pcds[i] = pcd

        pose_graph = self.full_registration(pcds)
        print("Optimizing PoseGraph")

        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=self.max_correspondence_distance_fine,
            edge_prune_threshold=0.25,
            reference_node=0,
        )
        o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option,
        )
        self.pose_graph = pose_graph

    def combine(self, pcds: List[o3d.geometry.PointCloud], downsample: bool = True) -> o3d.geometry.PointCloud:
        combined = o3d.geometry.PointCloud()
        for point_id in range(len(pcds)):
            combined += deepcopy(pcds[point_id]).transform(self.pose_graph.nodes[point_id].pose)
        if downsample:
            combined = combined.voxel_down_sample(self.voxel_size)
        return combined


# pcds = [pcds[2], pcds[3]]
# pcds = [pcds[0], pcds[1]]

for i, pcd in enumerate(pcds):
    np.savetxt(f"cam{i}.txt", pcd.points)

# for pcd in pcds:
#     o3d.visualization.draw_geometries([pcd])

# o3d.visualization.draw_geometries(pcds)

voxel_size = 0.01

pcds_down = []
for pcd in pcds:
    pcd_down = deepcopy(pcd).voxel_down_sample(voxel_size)
    pcd_down.estimate_normals()
    pcds_down.append(pcd_down)

o3d.visualization.draw_geometries(pcds_down)


matcher = PointCloudMatcher(
    voxel_size=0.01,
    max_correspondence_distance_coarse_scale=15.0,
    max_correspondence_distance_fine_scale=1.5,
)
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    matcher.fit(pcds_down)

print("Transform points and display")
# pcds_down = matcher.transform(pcds_down)
# pcds = matcher.transform(pcds)s
# o3d.visualization.draw_geometries(pcds_down)
# o3d.visualization.draw_geometries(pcds)
o3d.visualization.draw_geometries([matcher.combine(pcds, downsample=False)])


# matcher = PointCloudMatcher(
#     voxel_size=0.01,
#     max_correspondence_distance_coarse_scale=15.0,
#     max_correspondence_distance_fine_scale=1.5,
# )

# matcher.fit([pcds[0], pcds[1]])
# combine01 = matcher.combine([pcds[0], pcds[1]])

# matcher.fit([pcds[2], pcds[3]])
# combine23 = matcher.combine([pcds[2], pcds[3]])
# o3d.visualization.draw_geometries([combine01, combine23])


# matcher.fit([combine01, combine23])
# combine0123 = matcher.combine([combine01, combine23])
# o3d.visualization.draw_geometries([combine0123])
