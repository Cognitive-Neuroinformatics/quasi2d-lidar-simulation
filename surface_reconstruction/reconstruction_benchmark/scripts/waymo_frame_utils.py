#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import tensorflow as tf

from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.utils import frame_utils


def laser_name_to_enum(name: str) -> int:
    normalized = name.strip().upper()

    name_mapping = {
        "TOP": dataset_pb2.LaserName.TOP,
        "FRONT": dataset_pb2.LaserName.FRONT,
        "SIDE_LEFT": dataset_pb2.LaserName.SIDE_LEFT,
        "SIDE_RIGHT": dataset_pb2.LaserName.SIDE_RIGHT,
        "REAR": dataset_pb2.LaserName.REAR,
    }

    if normalized not in name_mapping:
        raise ValueError(
            f"Unknown LiDAR name '{name}'. "
            f"Valid names: {', '.join(name_mapping)}"
        )

    return name_mapping[normalized]


def laser_enum_to_name(value: int) -> str:
    value_mapping = {
        dataset_pb2.LaserName.TOP: "TOP",
        dataset_pb2.LaserName.FRONT: "FRONT",
        dataset_pb2.LaserName.SIDE_LEFT: "SIDE_LEFT",
        dataset_pb2.LaserName.SIDE_RIGHT: "SIDE_RIGHT",
        dataset_pb2.LaserName.REAR: "REAR",
    }

    return value_mapping.get(value, f"UNKNOWN_{value}")


def sensor_origin_from_calibration(
    calibration: dataset_pb2.LaserCalibration,
) -> np.ndarray:
    extrinsic = np.asarray(
        calibration.extrinsic.transform,
        dtype=np.float64,
    ).reshape(4, 4)

    return extrinsic[:3, 3]

def iterate_waymo_frames(
    tfrecord_path: Path,
) -> Iterator[tuple[int, dataset_pb2.Frame]]:
    dataset = tf.data.TFRecordDataset(
        str(tfrecord_path),
        compression_type="",
    )

    for frame_index, record in enumerate(dataset):
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(record.numpy()))

        yield frame_index, frame


def get_vehicle_pose(
    frame: dataset_pb2.Frame,
) -> np.ndarray:
    return np.asarray(
        frame.pose.transform,
        dtype=np.float64,
    ).reshape(4, 4)


def transform_xyz(
    xyz: np.ndarray,
    transform: np.ndarray,
) -> np.ndarray:
    homogeneous = np.concatenate(
        [
            xyz.astype(np.float64),
            np.ones((len(xyz), 1), dtype=np.float64),
        ],
        axis=1,
    )

    transformed = homogeneous @ transform.T
    return transformed[:, :3]


def crop_xyz(
    xyz: np.ndarray,
    crop_config: dict,
) -> tuple[np.ndarray, np.ndarray]:
    if not crop_config.get("enabled", False):
        return xyz, np.ones(len(xyz), dtype=bool)

    mask = (
        (xyz[:, 0] >= float(crop_config["x_min"]))
        & (xyz[:, 0] <= float(crop_config["x_max"]))
        & (xyz[:, 1] >= float(crop_config["y_min"]))
        & (xyz[:, 1] <= float(crop_config["y_max"]))
        & (xyz[:, 2] >= float(crop_config["z_min"]))
        & (xyz[:, 2] <= float(crop_config["z_max"]))
    )

    return xyz[mask], mask

def extract_lidar_points(
    frame: dataset_pb2.Frame,
    requested_lidars: list[str],
    return_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if return_index not in (0, 1):
        raise ValueError(
            "return_index must be 0 for first return or 1 for second return"
        )

    requested_enums = {
        laser_name_to_enum(name)
        for name in requested_lidars
    }

    (
        range_images,
        camera_projections,
        segmentation_labels,
        range_image_top_pose,
    ) = frame_utils.parse_range_image_and_camera_projection(frame)

    del segmentation_labels

    points_per_lidar, _ = frame_utils.convert_range_image_to_point_cloud(
        frame,
        range_images,
        camera_projections,
        range_image_top_pose,
        ri_index=return_index,
    )

    calibrations = sorted(
        frame.context.laser_calibrations,
        key=lambda calibration: calibration.name,
    )

    all_xyz: list[np.ndarray] = []
    all_intensity: list[np.ndarray] = []
    all_elongation: list[np.ndarray] = []
    all_origins: list[np.ndarray] = []

    for calibration, lidar_points in zip(
        calibrations,
        points_per_lidar,
    ):
        lidar_name = laser_enum_to_name(calibration.name)

        if calibration.name not in requested_enums:
            continue

        lidar_points = np.asarray(lidar_points)

        if lidar_points.ndim != 2 or lidar_points.shape[1] < 3:
            raise ValueError(
                f"Unexpected point shape for {lidar_name}: "
                f"{lidar_points.shape}"
            )

        xyz = lidar_points[:, :3].astype(np.float64)

        if lidar_points.shape[1] >= 4:
            intensity = lidar_points[:, 3].astype(np.float32)
        else:
            intensity = np.zeros(len(xyz), dtype=np.float32)

        if lidar_points.shape[1] >= 5:
            elongation = lidar_points[:, 4].astype(np.float32)
        else:
            elongation = np.zeros(len(xyz), dtype=np.float32)

        origin = sensor_origin_from_calibration(calibration)
        origins = np.repeat(
            origin[None, :],
            len(xyz),
            axis=0,
        )

        print(
            f"{lidar_name}: {len(xyz):,} points, "
            f"sensor origin = {origin}"
        )

        all_xyz.append(xyz)
        all_intensity.append(intensity)
        all_elongation.append(elongation)
        all_origins.append(origins)

    if not all_xyz:
        requested = ", ".join(requested_lidars)
        raise RuntimeError(
            f"No points extracted for requested LiDARs: {requested}"
        )

    return (
        np.concatenate(all_xyz, axis=0),
        np.concatenate(all_intensity, axis=0),
        np.concatenate(all_elongation, axis=0),
        np.concatenate(all_origins, axis=0),
    )