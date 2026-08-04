#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
import tensorflow as tf
import yaml

from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.utils import frame_utils


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML configuration: {path}")

    return config


def get_frame(tfrecord_path: Path, frame_index: int) -> dataset_pb2.Frame:
    """Read one Waymo frame from a TFRecord file."""
    if not tfrecord_path.exists():
        raise FileNotFoundError(f"TFRecord not found: {tfrecord_path}")

    dataset = tf.data.TFRecordDataset(
        str(tfrecord_path),
        compression_type="",
    )

    for index, record in enumerate(dataset):
        if index == frame_index:
            frame = dataset_pb2.Frame()
            frame.ParseFromString(record.numpy())
            return frame

    raise IndexError(
        f"Frame index {frame_index} does not exist in {tfrecord_path}"
    )

def laser_enum_to_name(value: int) -> str:
    """Convert a Waymo LiDAR enum value to a readable name."""
    value_mapping = {
        dataset_pb2.LaserName.TOP: "TOP",
        dataset_pb2.LaserName.FRONT: "FRONT",
        dataset_pb2.LaserName.SIDE_LEFT: "SIDE_LEFT",
        dataset_pb2.LaserName.SIDE_RIGHT: "SIDE_RIGHT",
        dataset_pb2.LaserName.REAR: "REAR",
    }

    return value_mapping.get(value, f"UNKNOWN_{value}")

def laser_name_to_enum(name: str) -> int:
    """Convert a LiDAR name such as TOP to its Waymo enum value."""
    normalized = name.strip().upper()

    name_mapping = {
        "TOP": dataset_pb2.LaserName.TOP,
        "FRONT": dataset_pb2.LaserName.FRONT,
        "SIDE_LEFT": dataset_pb2.LaserName.SIDE_LEFT,
        "SIDE_RIGHT": dataset_pb2.LaserName.SIDE_RIGHT,
        "REAR": dataset_pb2.LaserName.REAR,
    }

    if normalized not in name_mapping:
        valid_names = ", ".join(name_mapping)
        raise ValueError(
            f"Unknown LiDAR name '{name}'. Valid names: {valid_names}"
        )

    return name_mapping[normalized]

def sensor_origin_from_calibration(
    calibration: dataset_pb2.LaserCalibration,
) -> np.ndarray:
    """
    Return the LiDAR origin in the vehicle coordinate frame.

    The calibration extrinsic transforms points from the LiDAR coordinate
    system to the vehicle coordinate system. Its translation is therefore
    the LiDAR origin in vehicle coordinates.
    """
    extrinsic = np.asarray(
        calibration.extrinsic.transform,
        dtype=np.float64,
    ).reshape(4, 4)

    return extrinsic[:3, 3]


def extract_points(
    frame: dataset_pb2.Frame,
    requested_lidars: list[str],
    return_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract points from selected Waymo LiDARs.

    Returns:
        xyz:
            Shape (N, 3).
        intensity:
            Shape (N,).
        elongation:
            Shape (N,).
        sensor_origins:
            Shape (N, 3).
    """
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

        # Depending on the Waymo package version, the converted point array
        # may contain only XYZ. Intensity and elongation are therefore
        # initialized safely when unavailable.
        if lidar_points.shape[1] >= 4:
            intensity = lidar_points[:, 3].astype(np.float32)
        else:
            intensity = np.zeros(len(xyz), dtype=np.float32)

        if lidar_points.shape[1] >= 5:
            elongation = lidar_points[:, 4].astype(np.float32)
        else:
            elongation = np.zeros(len(xyz), dtype=np.float32)

        origin = sensor_origin_from_calibration(calibration)
        origins = np.repeat(origin[None, :], len(xyz), axis=0)

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


def remove_non_finite(
    xyz: np.ndarray,
    intensity: np.ndarray,
    elongation: np.ndarray,
    sensor_origins: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Remove rows containing NaN or infinity."""
    mask = (
        np.isfinite(xyz).all(axis=1)
        & np.isfinite(intensity)
        & np.isfinite(elongation)
        & np.isfinite(sensor_origins).all(axis=1)
    )

    return (
        xyz[mask],
        intensity[mask],
        elongation[mask],
        sensor_origins[mask],
    )


def crop_points(
    xyz: np.ndarray,
    intensity: np.ndarray,
    elongation: np.ndarray,
    sensor_origins: np.ndarray,
    crop_config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply an axis-aligned crop in vehicle coordinates."""
    if not crop_config.get("enabled", False):
        return xyz, intensity, elongation, sensor_origins

    mask = (
        (xyz[:, 0] >= float(crop_config["x_min"]))
        & (xyz[:, 0] <= float(crop_config["x_max"]))
        & (xyz[:, 1] >= float(crop_config["y_min"]))
        & (xyz[:, 1] <= float(crop_config["y_max"]))
        & (xyz[:, 2] >= float(crop_config["z_min"]))
        & (xyz[:, 2] <= float(crop_config["z_max"]))
    )

    return (
        xyz[mask],
        intensity[mask],
        elongation[mask],
        sensor_origins[mask],
    )


def intensity_to_colors(intensity: np.ndarray) -> np.ndarray:
    """Map intensity values to grayscale RGB for PLY visualization."""
    if intensity.size == 0:
        return np.empty((0, 3), dtype=np.float64)

    finite = intensity[np.isfinite(intensity)]

    if finite.size == 0:
        normalized = np.zeros_like(intensity, dtype=np.float64)
    else:
        lower = float(np.percentile(finite, 1))
        upper = float(np.percentile(finite, 99))

        if upper <= lower:
            normalized = np.zeros_like(intensity, dtype=np.float64)
        else:
            normalized = np.clip(
                (intensity - lower) / (upper - lower),
                0.0,
                1.0,
            )

    return np.repeat(normalized[:, None], 3, axis=1)


def save_outputs(
    output_directory: Path,
    xyz: np.ndarray,
    intensity: np.ndarray,
    elongation: np.ndarray,
    sensor_origins: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    """Save PLY, NPZ and JSON files."""
    output_directory.mkdir(parents=True, exist_ok=True)

    stem = (
        f"frame_{metadata['frame_index']:03d}_"
        f"{'_'.join(name.lower() for name in metadata['lidars'])}_"
        f"return_{metadata['return_number']}"
    )

    ply_path = output_directory / f"{stem}.ply"
    npz_path = output_directory / f"{stem}.npz"
    json_path = output_directory / f"{stem}_metadata.json"

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(xyz)
    cloud.colors = o3d.utility.Vector3dVector(
        intensity_to_colors(intensity)
    )

    success = o3d.io.write_point_cloud(
        str(ply_path),
        cloud,
        write_ascii=False,
        compressed=False,
        print_progress=True,
    )

    if not success:
        raise RuntimeError(f"Failed to write PLY file: {ply_path}")

    np.savez_compressed(
        npz_path,
        xyz=xyz.astype(np.float32),
        intensity=intensity.astype(np.float32),
        elongation=elongation.astype(np.float32),
        sensor_origins=sensor_origins.astype(np.float32),
    )

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("\nSaved:")
    print(f"  PLY:  {ply_path}")
    print(f"  NPZ:  {npz_path}")
    print(f"  JSON: {json_path}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract one Waymo LiDAR frame using a YAML config."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to single_frame.yaml",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    config = load_yaml(args.config)

    input_config = config["input"]
    crop_config = config.get("crop", {})
    preprocessing_config = config.get("preprocessing", {})

    tfrecord_path = Path(input_config["tfrecord"]).expanduser().resolve()
    frame_index = int(input_config.get("frame_index", 0))
    lidar_names = input_config.get("lidar_names", ["TOP"])
    return_index = int(input_config.get("return_index", 0))

    output_directory = Path("data/extracted")

    print(f"TFRecord:    {tfrecord_path}")
    print(f"Frame index: {frame_index}")
    print(f"LiDARs:      {lidar_names}")
    print(f"Return:      {return_index + 1}")

    frame = get_frame(tfrecord_path, frame_index)

    xyz, intensity, elongation, sensor_origins = extract_points(
        frame=frame,
        requested_lidars=lidar_names,
        return_index=return_index,
    )

    original_count = len(xyz)
    print(f"\nExtracted total: {original_count:,} points")

    if preprocessing_config.get("remove_non_finite", True):
        xyz, intensity, elongation, sensor_origins = remove_non_finite(
            xyz,
            intensity,
            elongation,
            sensor_origins,
        )
        print(f"After non-finite removal: {len(xyz):,} points")

    xyz, intensity, elongation, sensor_origins = crop_points(
        xyz,
        intensity,
        elongation,
        sensor_origins,
        crop_config,
    )

    print(f"After crop: {len(xyz):,} points")

    if len(xyz) == 0:
        raise RuntimeError(
            "No points remain after cropping. Check the crop limits."
        )

    metadata = {
        "source_tfrecord": str(tfrecord_path),
        "frame_index": frame_index,
        "timestamp_micros": int(frame.timestamp_micros),
        "context_name": frame.context.name,
        "coordinate_frame": "waymo_vehicle",
        "lidars": lidar_names,
        "return_index": return_index,
        "return_number": return_index + 1,
        "original_point_count": original_count,
        "saved_point_count": len(xyz),
        "crop": crop_config,
        "fields": [
            "xyz",
            "intensity",
            "elongation",
            "sensor_origins",
        ],
    }

    save_outputs(
        output_directory=output_directory,
        xyz=xyz,
        intensity=intensity,
        elongation=elongation,
        sensor_origins=sensor_origins,
        metadata=metadata,
    )

    return 0


if __name__ == "__main__":
    main()