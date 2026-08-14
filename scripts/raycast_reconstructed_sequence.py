import argparse
import os

import numpy as np
import yaml

from reconstruction.reconstructed_scene_loader import (
    ReconstructedSceneLoader,
)

from pointcloud_transformer.pointcloud_transformer import (
    PointCloudTransformer,
)


POINT_CLOUD_RANGE = [
    -75.2, -75.2, -2.0,
     75.2,  75.2,  4.0,
]


def find_sensor_config(scala2_cfg, sensor_name):
    for sensor_cfg in scala2_cfg["positions"]:
        if sensor_cfg["name"] == sensor_name:
            return sensor_cfg

    raise ValueError(
        f"Sensor '{sensor_name}' not found in Scala2 config."
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scene-dir",
        required=True,
    )

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--sensor",
        default="front_center",
    )

    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--end-frame",
        type=int,
        default=198,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
    )

    args = parser.parse_args()

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    # =========================================================
    # CONFIG
    # =========================================================

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    scala2_cfg = cfg["sensors"]["scala2"]
    common = scala2_cfg["common"]

    sensor_cfg = find_sensor_config(
        scala2_cfg,
        args.sensor,
    )

    voxel_size = float(
        cfg["bev"]["voxel_size"]
    )

    print("Voxel size:", voxel_size)

    # =========================================================
    # SENSOR POSE
    # =========================================================

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

    sensor_metadata = {
        "start_point": start_point,
        "rotation_quaternion_xyzw": quaternion_xyzw,

        # fallback field for legacy paths;
        # full quaternion will actually be used
        "rotation_angle": 0.0,
    }

    # =========================================================
    # LOAD RECONSTRUCTION ONCE
    # =========================================================

    loader = ReconstructedSceneLoader(
        args.scene_dir
    )

    transformer = PointCloudTransformer(
        voxel_size=voxel_size,
        ground_removal_method=None,
    )

    # Full-quaternion vehicle -> sensor transform.
    _, T_sensor_from_vehicle = (
        transformer.get_sensor_transforms(
            sensor_metadata,
            bev_padding=None,
        )
    )

    print()
    print("Sensor:", args.sensor)
    print("Origin in vehicle:", start_point)
    print("Quaternion xyzw:", quaternion_xyzw)
    print()

    # =========================================================
    # PROCESS FRAMES
    # =========================================================

    for frame_idx in range(
        args.start_frame,
        args.end_frame + 1,
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
                f"[{frame_idx:03d}] exists -> skipping"
            )
            continue

        print()
        print("=" * 70)
        print(f"FRAME {frame_idx:03d}")
        print("=" * 70)

        # -----------------------------------------------------
        # 1. Complete reconstructed scene in VEHICLE frame
        # -----------------------------------------------------

        frame = loader.load_frame_vehicle(
            frame_idx=frame_idx,
            point_cloud_range=POINT_CLOUD_RANGE,
        )

        print("Frame keys:", frame.keys())

        for key, value in frame.items():
            if isinstance(value, np.ndarray):
                print(
                    f"{key}: shape={value.shape}, dtype={value.dtype}"
                )
                
                xyz_vehicle = frame["xyz"].astype(
            np.float32
        )

        print(
            "Input reconstructed points:",
            f"{len(xyz_vehicle):,}"
        )

        # -----------------------------------------------------
        # 2. Add dummy intensity
        #
        # Existing Bresenham code expects at least XYZ[I].
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # 3. Scala2 mirror state
        # -----------------------------------------------------

        mirror_side = frame_idx % 2

        print(
            "Mirror side:",
            mirror_side
        )

        # -----------------------------------------------------
        # 4. ACTUAL SCALA2 + BRESENHAM RAYCAST
        #
        # This performs:
        #
        # create_voxel_map()
        # calculate_endpoints_vectorized_scala2()
        # bresenham3D()
        # first occupied voxel
        # representative point selection
        # projection onto exact ray
        #
        # Output here is still VEHICLE frame.
        # -----------------------------------------------------

        projection_method = common.get(
            "projection_method",
            "same_distance",
        )

        print(
            "Projection method:",
            projection_method,
        )

        simulated_vehicle, ray_metadata = (
            transformer.transform_point_cloud(
                sensor="scala2",

                original_pointcloud=reconstructed_pc,

                start_point=start_point,

                dist=float(
                    common["distance"]
                ),

                horizontal_angle_min=float(
                    common["horizontal_angle_min"]
                ),

                horizontal_angle_max=float(
                    common["horizontal_angle_max"]
                ),

                horizontal_rays=None,
                vertical_angles=None,
                rotation_angle=None,

                rotation_quaternion_xyzw=(
                    quaternion_xyzw
                ),

                mirror_side=mirror_side,

                scala2_common=common,

                projection_method=projection_method,
            )
        )

        print(
            "Bresenham returns:",
            f"{len(simulated_vehicle):,}"
        )

        # -----------------------------------------------------
        # 5. VEHICLE -> selected Scala2 SENSOR frame
        # -----------------------------------------------------

        if len(simulated_vehicle) > 0:

            simulated_sensor = (
                transformer.transform_pc_to_sensor_frame(
                    simulated_vehicle,
                    sensor_metadata,
                    bev_padding=None,
                )
            ).astype(np.float32)

        else:

            simulated_sensor = np.empty(
                (0, 4),
                dtype=np.float32,
            )

        print(
            "Sensor-frame returns:",
            f"{len(simulated_sensor):,}"
        )

        # -----------------------------------------------------
        # 6. Save
        # -----------------------------------------------------

        np.savez_compressed(
            output_path,

            # x y z intensity in Scala2 sensor frame
            points=simulated_sensor,

            xyz=simulated_sensor[:, :3],

            frame_idx=np.int32(
                frame_idx
            ),

            mirror_side=np.int8(
                mirror_side
            ),

            sensor_name=np.asarray(
                args.sensor
            ),

            projection_method=np.asarray(
                projection_method
            ),

            vehicle_to_sensor=(
                T_sensor_from_vehicle
            ),

            vehicle_to_world=(
                frame["vehicle_to_world"]
            ),

            azimuth_deg_sensor=(
                ray_metadata[
                    "azimuth_deg_sensor"
                ]
            ),

            elevation_deg_sensor=(
                ray_metadata[
                    "elevation_deg_sensor"
                ]
            ),

            apd_group=(
                ray_metadata["apd_group"]
            ),

            layer=(
                ray_metadata["layer"]
            ),

            column_id=(
                ray_metadata["column_id"]
            ),
        )
        print(
            "Saved:",
            output_path
        )


if __name__ == "__main__":
    main()