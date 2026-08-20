import argparse
import os

import numpy as np
import open3d as o3d
import yaml

from reconstruction.reconstructed_scene_loader import (
    ReconstructedSceneLoader,
)
from pointcloud_transformer.pointcloud_transformer import (
    PointCloudTransformer,
)


POINT_CLOUD_RANGE = [
    -75.2,
    -75.2,
    -2.0,
    75.2,
    75.2,
    4.0,
]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--frame",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--sensor",
        type=str,
        default="front_center",
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--scene-dir",
        type=str,
        default=(
            "/home/samanti/Documents/Uni_Bremen/PhD/"
            "my_workspace/lidar_surface_reconstruction/data/"
            "scene_merger_outputs/full_scene"
        ),
    )

    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Load config
    # ---------------------------------------------------------

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    scala2_cfg = cfg["sensors"]["scala2"]

    common = scala2_cfg["common"]

    sensor_cfg = None

    for sensor in scala2_cfg["positions"]:
        if sensor["name"] == args.sensor:
            sensor_cfg = sensor
            break

    if sensor_cfg is None:
        raise ValueError(
            f"Sensor '{args.sensor}' not found."
        )

    # ---------------------------------------------------------
    # Load reconstructed scene in vehicle frame
    # ---------------------------------------------------------

    loader = ReconstructedSceneLoader(
        args.scene_dir
    )

    frame = loader.load_frame_vehicle(
        frame_idx=args.frame,
        point_cloud_range=POINT_CLOUD_RANGE,
    )

    xyz_vehicle = frame["xyz"].astype(
        np.float32
    )

    print()
    print("Reconstructed scene")
    print("Frame:", args.frame)
    print("Vehicle-frame points:", len(xyz_vehicle))
    print("min:", xyz_vehicle.min(axis=0))
    print("max:", xyz_vehicle.max(axis=0))

    # ---------------------------------------------------------
    # Add dummy intensity
    # ---------------------------------------------------------

    intensity = np.zeros(
        (len(xyz_vehicle), 1),
        dtype=np.float32,
    )

    reconstructed_pc = np.concatenate(
        [
            xyz_vehicle,
            intensity,
        ],
        axis=1,
    )

    # ---------------------------------------------------------
    # Build Scala2 transformer
    # ---------------------------------------------------------

    voxel_size = float(cfg["bev"]["voxel_size"])

    print("Voxel size:", voxel_size)

    transformer = PointCloudTransformer(
        voxel_size=voxel_size,
        ground_removal_method=None,
    )



    start_point = tuple(
        sensor_cfg["start_point"]
    )

    quaternion_xyzw = (
        float(sensor_cfg["rotation_angle_x"]),
        float(sensor_cfg["rotation_angle_y"]),
        float(sensor_cfg["rotation_angle_z"]),
        float(sensor_cfg["rotation_angle_w"]),
    )

    # Same mirror convention as your existing code
    mirror_side = args.frame % 2

    # ---------------------------------------------------------
    # Raycast in VEHICLE frame
    # ---------------------------------------------------------

    simulated_vehicle, ray_metadata = (
        transformer.transform_point_cloud(
            sensor="scala2",
            original_pointcloud=reconstructed_pc,
            start_point=start_point,
            dist=float(common["distance"]),
            horizontal_angle_min=float(
                common["horizontal_angle_min"]
            ),
            horizontal_angle_max=float(
                common["horizontal_angle_max"]
            ),
            horizontal_rays=None,
            vertical_angles=None,
            rotation_angle=None,
            rotation_quaternion_xyzw=quaternion_xyzw,
            mirror_side=mirror_side,
            scala2_common=common,
        )
    )

    print()
    print("Raycast result in vehicle frame:")
    print("Points:", len(simulated_vehicle))

    if len(simulated_vehicle) == 0:
        raise RuntimeError(
            "No Scala2 ray hits found."
        )

    # ---------------------------------------------------------
    # Transform VEHICLE -> selected SENSOR frame
    # ---------------------------------------------------------

    metadata = {
        "start_point": start_point,
        "rotation_quaternion_xyzw": quaternion_xyzw,
        "rotation_angle": 0.0,
    }

    simulated_sensor = (
        transformer.transform_pc_to_sensor_frame(
            simulated_vehicle,
            metadata,
            bev_padding=None,
        )
    )

    print()
    print("Final sensor-frame cloud:")
    print("Sensor:", args.sensor)
    print("Points:", len(simulated_sensor))
    print("min:", simulated_sensor[:, :3].min(axis=0))
    print("max:", simulated_sensor[:, :3].max(axis=0))

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    if args.output is None:
        args.output = (
            f"frame_{args.frame:03d}_"
            f"{args.sensor}_raycast.npy"
        )

    np.save(
        args.output,
        simulated_sensor.astype(np.float32),
    )

    print()
    print("Saved:")
    print(args.output)

    # ---------------------------------------------------------
    # Open3D visualization in SENSOR coordinates
    # ---------------------------------------------------------

    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(
        simulated_sensor[:, :3].astype(
            np.float64
        )
    )

    coordinate_frame = (
        o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=2.0,
            origin=[0, 0, 0],
        )
    )

    o3d.visualization.draw_geometries(
        [
            pcd,
            coordinate_frame,
        ],
        window_name=(
            f"Scala2 {args.sensor} | "
            f"frame {args.frame:03d}"
        ),
        width=1600,
        height=900,
    )


if __name__ == "__main__":
    main()