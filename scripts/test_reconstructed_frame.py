import numpy as np
import open3d as o3d

from reconstruction.reconstructed_scene_loader import (
    ReconstructedSceneLoader,
)


SCENE_DIR = (
    "/home/samanti/Documents/Uni_Bremen/PhD/"
    "my_workspace/lidar_surface_reconstruction/data/"
    "scene_merger_outputs/full_scene"
)


POINT_CLOUD_RANGE = [
    -75.2,
    -75.2,
    -2.0,
    75.2,
    75.2,
    4.0,
]


loader = ReconstructedSceneLoader(
    SCENE_DIR
)

frame = loader.load_frame_vehicle(
    frame_idx=0,
    point_cloud_range=POINT_CLOUD_RANGE,
)

xyz = frame["xyz"]

print()
print("Frame 000")
print("Vehicle-frame points:", len(xyz))
print("min:", xyz.min(axis=0))
print("max:", xyz.max(axis=0))
print("mean:", xyz.mean(axis=0))


pcd = o3d.geometry.PointCloud()

pcd.points = o3d.utility.Vector3dVector(
    xyz.astype(np.float64)
)

o3d.visualization.draw_geometries(
    [pcd],
    window_name="Frame 000 reconstructed scene - vehicle frame",
    width=1600,
    height=900,
)