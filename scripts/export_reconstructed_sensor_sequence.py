# scripts/export_reconstructed_sensor_sequence.py

import argparse
import os

import numpy as np
import yaml

from reconstruction.reconstructed_scene_loader import ReconstructedSceneLoader
from pointcloud_transformer.pointcloud_transformer import PointCloudTransformer


POINT_CLOUD_RANGE = [
    -75.2, -75.2, -2.0,
     75.2,  75.2,  4.0,
]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--scene-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--sensor", default="front_center")

    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=198)

    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--skip-existing", action="store_true")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---------------------------------------------------------
    # Config
    # ---------------------------------------------------------

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    scala2_cfg = cfg["sensors"]["scala2"]

    sensor_cfg = None

    for sensor in scala2_cfg["positions"]:
        if sensor["name"] == args.sensor:
            sensor_cfg = sensor
            break

    if sensor_cfg is None:
        raise ValueError(
            f"Sensor '{args.sensor}' not found."
        )

    start_point = np.asarray(
        sensor_cfg["start_point"],
        dtype=np.float32,
    )

    quaternion_xyzw = (
        float(sensor_cfg["rotation_angle_x"]),
        float(sensor_cfg["rotation_angle_y"]),
        float(sensor_cfg["rotation_angle_z"]),
        float(sensor_cfg["rotation_angle_w"]),
    )

    # ---------------------------------------------------------
    # Scene loader
    # ---------------------------------------------------------

    loader = ReconstructedSceneLoader(
        args.scene_dir
    )

    # Transformer is only used here for the sensor transform.
    transformer = PointCloudTransformer(
        voxel_size=float(cfg["bev"]["voxel_size"]),
        ground_removal_method=None,
    )

    metadata = {
        "start_point": start_point,
        "rotation_quaternion_xyzw": quaternion_xyzw,
        "rotation_angle": 0.0,
    }

    _, T_sensor_from_vehicle = (
        transformer.get_sensor_transforms(
            metadata,
            bev_padding=None,
        )
    )

    # ---------------------------------------------------------
    # Frames
    # ---------------------------------------------------------

    for frame_idx in range(
        args.start_frame,
        args.end_frame + 1
    ):

        output_path = os.path.join(
            args.output_dir,
            f"frame_{frame_idx:03d}.npz",
        )

        if (
            args.skip_existing
            and os.path.exists(output_path)
        ):
            print(
                f"[{frame_idx:03d}] exists - skipping"
            )
            continue

        # Complete reconstructed scene in VEHICLE frame:
        frame = loader.load_frame_vehicle(
            frame_idx=frame_idx,
            point_cloud_range=POINT_CLOUD_RANGE,
        )

        xyz_vehicle = frame["xyz"].astype(
            np.float32
        )

        # -----------------------------------------------------
        # VEHICLE -> Scala2 sensor
        # -----------------------------------------------------

        ones = np.ones(
            (len(xyz_vehicle), 1),
            dtype=np.float32,
        )

        xyz_vehicle_h = np.concatenate(
            [xyz_vehicle, ones],
            axis=1,
        )

        xyz_sensor = (
            xyz_vehicle_h
            @ T_sensor_from_vehicle.T
        )[:, :3].astype(np.float32)

        # -----------------------------------------------------
        # Optional second crop in SENSOR frame
        #
        # Keeps the saved local sensor scene bounded.
        # -----------------------------------------------------

        r = np.asarray(
            POINT_CLOUD_RANGE,
            dtype=np.float32,
        )

        mask = (
            (xyz_sensor[:, 0] >= r[0])
            & (xyz_sensor[:, 0] <= r[3])
            & (xyz_sensor[:, 1] >= r[1])
            & (xyz_sensor[:, 1] <= r[4])
            & (xyz_sensor[:, 2] >= r[2])
            & (xyz_sensor[:, 2] <= r[5])
        )

        xyz_sensor = xyz_sensor[mask]

        # -----------------------------------------------------
        # Save
        # -----------------------------------------------------

        np.savez_compressed(
            output_path,
            xyz=xyz_sensor,
            frame_idx=np.int32(frame_idx),
            sensor_name=args.sensor,
            sensor_to_vehicle=np.linalg.inv(
                T_sensor_from_vehicle
            ).astype(np.float32),
            vehicle_to_sensor=T_sensor_from_vehicle,
            vehicle_to_world=frame["vehicle_to_world"],
        )

        print(
            f"[{frame_idx:03d}] "
            f"{len(xyz_sensor):,} points -> "
            f"{output_path}"
        )


if __name__ == "__main__":
    main()