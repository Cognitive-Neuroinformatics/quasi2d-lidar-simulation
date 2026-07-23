#!/usr/bin/env python3
"""
Comprehensive Waymo Open Dataset LiDAR range-image debugger.

For one TFRecord frame, this script:
  1. Parses every available LiDAR (TOP, FRONT, SIDE_LEFT, SIDE_RIGHT, REAR).
  2. Parses both returns independently.
  3. Converts each return from range-image coordinates to Cartesian XYZ.
  4. Saves per-return visualizations and NumPy arrays.
  5. Combines return 1 and return 2 for every LiDAR.
  6. Saves combined-return visualizations, point clouds, masks, and metadata.
  7. Saves beam-inclination and calibration diagnostics for each LiDAR.
  8. Builds frame-level summary files (TXT, CSV, JSON).
  9. Saves one frame-level point cloud containing all LiDARs and both returns.

Important interpretation
------------------------
A Waymo range-image pixel stores at least:
    channel 0: range
    channel 1: intensity
    channel 2: elongation
    channel 3: additional Waymo field (dataset/API-version dependent)

The H x W range-image grid is retained during Cartesian conversion:
    H x W x C polar image -> H x W x 3 Cartesian image
Only after masking range <= 0 are valid pixels gathered into an N x 3 point cloud.

Example
-------
python understand_waymo_range_images_all_lidars.py \
    --tfrecord /path/to/segment.tfrecord \
    --frame-index 0 \
    --lidar ALL \
    --returns BOTH \
    --save-dir range_image_debug_output
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils


LIDAR_NAMES = ["TOP", "FRONT", "SIDE_LEFT", "SIDE_RIGHT", "REAR"]
RETURN_LABELS = {0: "return_1", 1: "return_2"}

# laser enum -> {0: first return MatrixFloat, 1: second return MatrixFloat}
RangeImages = Dict[int, Dict[int, dataset_pb2.MatrixFloat]]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect all Waymo LiDAR range images, both returns, their Cartesian "
            "point clouds, and combined-return behavior."
        )
    )
    parser.add_argument(
        "--tfrecord",
        type=Path,
        required=True,
        help="Path to one Waymo .tfrecord file.",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="Zero-based frame index to inspect.",
    )
    parser.add_argument(
        "--lidar",
        type=str,
        default="ALL",
        choices=["ALL", *LIDAR_NAMES],
        help="Inspect one LiDAR or all five LiDARs.",
    )
    parser.add_argument(
        "--returns",
        type=str,
        default="BOTH",
        choices=["FIRST", "SECOND", "BOTH"],
        help="Returns to process. BOTH also creates combined-return outputs.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("range_image_debug_output"),
        help="Root output directory.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Resolution for saved figures.",
    )
    parser.add_argument(
        "--intensity-scale",
        type=float,
        default=1.0,
        help=(
            "Scale s for arctangent intensity display: "
            "(2/pi) * atan(max(intensity,0)/s)."
        ),
    )
    parser.add_argument(
        "--save-comparison-panels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save side-by-side return-1/return-2/difference panels.",
    )
    parser.add_argument(
        "--save-ply",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save ASCII PLY point clouds without requiring Open3D.",
    )
    parser.add_argument(
        "--visualize-open3d",
        action="store_true",
        help=(
            "Open an Open3D window. To avoid many windows, this is only used "
            "when a single LiDAR is selected."
        ),
    )
    return parser.parse_args()


def load_frame(tfrecord_path: Path, frame_index: int) -> dataset_pb2.Frame:
    if not tfrecord_path.exists():
        raise FileNotFoundError(f"TFRecord does not exist: {tfrecord_path}")
    if frame_index < 0:
        raise ValueError("frame-index must be non-negative.")

    dataset = tf.data.TFRecordDataset(str(tfrecord_path), compression_type="")
    for current_index, record in enumerate(dataset):
        if current_index != frame_index:
            continue
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(record.numpy()))

        print("\nLoaded frame")
        print("------------")
        print(f"TFRecord:             {tfrecord_path}")
        print(f"Frame index:          {frame_index}")
        print(f"Timestamp [us]:       {frame.timestamp_micros}")
        print(f"Number LiDAR sensors: {len(frame.lasers)}")
        print(f"Number camera images: {len(frame.images)}")
        return frame

    raise IndexError(f"Frame index {frame_index} was not found in {tfrecord_path}")


def lidar_name_to_enum(lidar_name: str) -> int:
    try:
        return dataset_pb2.LaserName.Name.Value(lidar_name)
    except ValueError as exc:
        raise ValueError(f"Unknown LiDAR name: {lidar_name}") from exc


def lidar_enum_to_name(lidar_enum: int) -> str:
    return dataset_pb2.LaserName.Name.Name(lidar_enum)


def requested_lidars(name: str) -> List[str]:
    return LIDAR_NAMES if name == "ALL" else [name]


def requested_return_indices(mode: str) -> List[int]:
    if mode == "FIRST":
        return [0]
    if mode == "SECOND":
        return [1]
    return [0, 1]


def print_available_lidars(frame: dataset_pb2.Frame) -> None:
    print("\nLiDARs stored in this frame")
    print("---------------------------")
    for laser in sorted(frame.lasers, key=lambda item: item.name):
        name = lidar_enum_to_name(laser.name)
        first_available = len(laser.ri_return1.range_image_compressed) > 0
        second_available = len(laser.ri_return2.range_image_compressed) > 0
        print(
            f"{name:12s} enum={laser.name:<2d} "
            f"first_return={first_available} second_return={second_available}"
        )


def decode_matrix_float(compressed: bytes) -> dataset_pb2.MatrixFloat:
    decoded = tf.io.decode_compressed(compressed, compression_type="ZLIB")
    matrix = dataset_pb2.MatrixFloat()
    matrix.ParseFromString(bytearray(decoded.numpy()))
    return matrix


def parse_range_images(
    frame: dataset_pb2.Frame,
) -> Tuple[RangeImages, Optional[dataset_pb2.MatrixFloat]]:
    """Parse all available first and second return range images."""
    range_images: RangeImages = {}
    range_image_top_pose: Optional[dataset_pb2.MatrixFloat] = None

    for laser in sorted(frame.lasers, key=lambda item: item.name):
        laser_name = lidar_enum_to_name(laser.name)
        range_images[laser.name] = {}
        print(f"\nParsing {laser_name}")

        first_compressed = laser.ri_return1.range_image_compressed
        if len(first_compressed) > 0:
            first = decode_matrix_float(first_compressed)
            range_images[laser.name][0] = first
            print(f"  first return shape:  {list(first.shape.dims)}")
            print(f"  flattened values:    {len(first.data)}")

            if laser.name == dataset_pb2.LaserName.TOP:
                pose_compressed = laser.ri_return1.range_image_pose_compressed
                if len(pose_compressed) > 0:
                    range_image_top_pose = decode_matrix_float(pose_compressed)
                    print(
                        "  TOP pixel-pose shape:",
                        list(range_image_top_pose.shape.dims),
                    )

        second_compressed = laser.ri_return2.range_image_compressed
        if len(second_compressed) > 0:
            second = decode_matrix_float(second_compressed)
            range_images[laser.name][1] = second
            print(f"  second return shape: {list(second.shape.dims)}")

    return range_images, range_image_top_pose


def matrix_float_to_tensor(matrix: dataset_pb2.MatrixFloat) -> tf.Tensor:
    return tf.reshape(
        tf.convert_to_tensor(matrix.data, dtype=tf.float32),
        matrix.shape.dims,
    )


def find_laser_calibration(
    frame: dataset_pb2.Frame,
    lidar_enum: int,
) -> dataset_pb2.LaserCalibration:
    for calibration in frame.context.laser_calibrations:
        if calibration.name == lidar_enum:
            return calibration
    raise KeyError(f"No calibration found for LiDAR {lidar_enum_to_name(lidar_enum)}")


def get_beam_inclinations(
    calibration: dataset_pb2.LaserCalibration,
    range_image_height: int,
) -> Tuple[tf.Tensor, Dict[str, Any]]:
    """Return row-ordered inclinations plus diagnostic metadata."""
    explicit_count = len(calibration.beam_inclinations)
    if explicit_count == 0:
        inclinations = range_image_utils.compute_inclination(
            tf.constant(
                [
                    calibration.beam_inclination_min,
                    calibration.beam_inclination_max,
                ],
                dtype=tf.float32,
            ),
            height=range_image_height,
        )
        source = "computed_from_min_max"
    else:
        inclinations = tf.constant(calibration.beam_inclinations, dtype=tf.float32)
        source = "explicit_calibration_array"

    # Match range-image row ordering used by the official Waymo conversion utility.
    inclinations = tf.reverse(inclinations, axis=[-1])
    beam_deg = np.rad2deg(inclinations.numpy())
    spacing_deg = np.diff(beam_deg)

    metadata: Dict[str, Any] = {
        "source": source,
        "explicit_inclination_count": explicit_count,
        "range_image_height": range_image_height,
        "beam_count": int(beam_deg.size),
        "maximum_elevation_deg": float(np.max(beam_deg)),
        "minimum_elevation_deg": float(np.min(beam_deg)),
        "vertical_fov_deg": float(np.max(beam_deg) - np.min(beam_deg)),
        "row_0_elevation_deg": float(beam_deg[0]),
        "last_row_elevation_deg": float(beam_deg[-1]),
        "spacing_mean_abs_deg": (
            float(np.mean(np.abs(spacing_deg))) if spacing_deg.size else math.nan
        ),
        "spacing_min_abs_deg": (
            float(np.min(np.abs(spacing_deg))) if spacing_deg.size else math.nan
        ),
        "spacing_max_abs_deg": (
            float(np.max(np.abs(spacing_deg))) if spacing_deg.size else math.nan
        ),
        "spacing_std_abs_deg": (
            float(np.std(np.abs(spacing_deg))) if spacing_deg.size else math.nan
        ),
        "angles_deg": beam_deg.tolist(),
    }

    lidar_name = lidar_enum_to_name(calibration.name)
    print("\nBeam inclination debug")
    print("----------------------")
    print(f"LiDAR:                 {lidar_name}")
    print(f"Source:                {source}")
    print(f"Range image height:    {range_image_height}")
    print(f"Explicit inclinations: {explicit_count}")
    print(f"Beam array shape:      {beam_deg.shape}")
    print(f"Minimum angle:         {beam_deg.min():.6f} deg")
    print(f"Maximum angle:         {beam_deg.max():.6f} deg")
    print(f"Vertical FOV:          {metadata['vertical_fov_deg']:.6f} deg")
    print(f"First 10 angles:       {beam_deg[:10]}")
    print(f"Last 10 angles:        {beam_deg[-10:]}")
    print(f"First 10 spacings:     {spacing_deg[:10]}")

    return inclinations, metadata


def build_top_pixel_pose(
    range_image_top_pose: dataset_pb2.MatrixFloat,
) -> tf.Tensor:
    top_pose_tensor = tf.reshape(
        tf.convert_to_tensor(range_image_top_pose.data, dtype=tf.float32),
        range_image_top_pose.shape.dims,
    )
    rotation = transform_utils.get_rotation_matrix(
        top_pose_tensor[..., 0],
        top_pose_tensor[..., 1],
        top_pose_tensor[..., 2],
    )
    translation = top_pose_tensor[..., 3:]
    return transform_utils.get_transform(rotation, translation)


def top_pixel_pose_statistics(
    range_image_top_pose: Optional[dataset_pb2.MatrixFloat],
) -> Optional[Dict[str, Any]]:
    if range_image_top_pose is None:
        return None
    pose = np.asarray(range_image_top_pose.data, dtype=np.float32).reshape(
        range_image_top_pose.shape.dims
    )
    names = ["roll_rad", "pitch_rad", "yaw_rad", "x_m", "y_m", "z_m"]
    result: Dict[str, Any] = {"shape": list(pose.shape)}
    for index, name in enumerate(names):
        values = pose[..., index]
        result[name] = {
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
            "std": float(values.std()),
        }
    return result


def convert_range_image_to_cartesian(
    frame: dataset_pb2.Frame,
    lidar_enum: int,
    range_image_tensor: tf.Tensor,
    range_image_top_pose: Optional[dataset_pb2.MatrixFloat],
) -> Tuple[tf.Tensor, tf.Tensor, np.ndarray, Dict[str, Any]]:
    calibration = find_laser_calibration(frame, lidar_enum)
    height = int(range_image_tensor.shape[0])
    beam_inclinations, beam_metadata = get_beam_inclinations(calibration, height)

    extrinsic = np.asarray(
        calibration.extrinsic.transform,
        dtype=np.float32,
    ).reshape(4, 4)
    frame_pose = np.asarray(frame.pose.transform, dtype=np.float32).reshape(4, 4)

    pixel_pose = None
    frame_pose_batch = None
    if lidar_enum == dataset_pb2.LaserName.TOP:
        if range_image_top_pose is None:
            raise ValueError("TOP LiDAR requires range_image_top_pose.")
        pixel_pose = tf.expand_dims(build_top_pixel_pose(range_image_top_pose), axis=0)
        frame_pose_batch = tf.expand_dims(tf.convert_to_tensor(frame_pose), axis=0)

    cartesian_batch = range_image_utils.extract_point_cloud_from_range_image(
        tf.expand_dims(range_image_tensor[..., 0], axis=0),
        tf.expand_dims(tf.convert_to_tensor(extrinsic), axis=0),
        tf.expand_dims(beam_inclinations, axis=0),
        pixel_pose=pixel_pose,
        frame_pose=frame_pose_batch,
    )
    cartesian = tf.squeeze(cartesian_batch, axis=0)
    return cartesian, beam_inclinations, extrinsic, beam_metadata


def form_point_cloud(
    range_image_tensor: tf.Tensor,
    cartesian_range_image: tf.Tensor,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid_mask = range_image_tensor[..., 0] > 0
    valid_indices = tf.where(valid_mask)
    points = tf.gather_nd(cartesian_range_image, valid_indices)
    polar_features = tf.gather_nd(range_image_tensor[..., 0:3], valid_indices)
    return points.numpy(), polar_features.numpy(), valid_indices.numpy()


def safe_percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if values.size else math.nan


def describe_values(values: np.ndarray) -> Dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "min": math.nan,
            "max": math.nan,
            "mean": math.nan,
            "median": math.nan,
            "p01": math.nan,
            "p05": math.nan,
            "p25": math.nan,
            "p75": math.nan,
            "p95": math.nan,
            "p99": math.nan,
            "p995": math.nan,
        }
    return {
        "min": float(finite.min()),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "p01": safe_percentile(finite, 1),
        "p05": safe_percentile(finite, 5),
        "p25": safe_percentile(finite, 25),
        "p75": safe_percentile(finite, 75),
        "p95": safe_percentile(finite, 95),
        "p99": safe_percentile(finite, 99),
        "p995": safe_percentile(finite, 99.5),
    }


def range_image_statistics(
    range_image_tensor: tf.Tensor,
    lidar_name: str,
    return_index: int,
) -> Dict[str, Any]:
    array = range_image_tensor.numpy()
    ranges = array[..., 0]
    intensity = array[..., 1]
    elongation = array[..., 2]
    valid_mask = ranges > 0

    valid_ranges = ranges[valid_mask]
    valid_intensity = intensity[valid_mask & np.isfinite(intensity)]
    valid_elongation = elongation[valid_mask & np.isfinite(elongation)]

    stats: Dict[str, Any] = {
        "lidar": lidar_name,
        "return_index": return_index,
        "return_name": RETURN_LABELS[return_index],
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "height": int(array.shape[0]),
        "width": int(array.shape[1]),
        "channels": int(array.shape[2]),
        "total_pixels": int(ranges.size),
        "valid_returns": int(valid_mask.sum()),
        "invalid_returns": int((~valid_mask).sum()),
        "valid_percentage": float(100.0 * valid_mask.mean()),
        "range_m": describe_values(valid_ranges),
        "intensity": describe_values(valid_intensity),
        "elongation": describe_values(valid_elongation),
        "channel_3": (
            describe_values(array[..., 3][valid_mask]) if array.shape[-1] > 3 else None
        ),
    }

    print("\nRange-image statistics")
    print("----------------------")
    print(f"LiDAR:                {lidar_name}")
    print(f"Return:               {return_index + 1}")
    print(f"Tensor shape:         {array.shape}")
    print(f"Total pixels/rays:    {stats['total_pixels']}")
    print(f"Valid returns:        {stats['valid_returns']}")
    print(f"Invalid returns:      {stats['invalid_returns']}")
    print(f"Return percentage:    {stats['valid_percentage']:.2f}%")
    if valid_ranges.size:
        print(f"Minimum range:        {stats['range_m']['min']:.3f} m")
        print(f"Maximum range:        {stats['range_m']['max']:.3f} m")
        print(f"Mean range:           {stats['range_m']['mean']:.3f} m")
    print(f"Intensity median:     {stats['intensity']['median']:.6f}")
    print(f"Intensity p99.5:      {stats['intensity']['p995']:.6f}")
    print(f"Intensity maximum:    {stats['intensity']['max']:.6f}")
    return stats


def arctan_intensity(
    intensity: np.ndarray,
    valid_mask: np.ndarray,
    scale: float,
) -> np.ndarray:
    if scale <= 0:
        raise ValueError("intensity-scale must be positive.")
    output = np.zeros_like(intensity, dtype=np.float32)
    mask = valid_mask & np.isfinite(intensity) & (intensity >= 0)
    output[mask] = (2.0 / np.pi) * np.arctan(intensity[mask] / scale)
    return output


def log_range(ranges: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    output = np.zeros_like(ranges, dtype=np.float32)
    output[valid_mask] = np.log1p(ranges[valid_mask])
    return output


def save_image(
    image: np.ndarray,
    title: str,
    output_path: Path,
    dpi: int,
    xlabel: str = "Range-image column / azimuth sample",
    ylabel: str = "Range-image row / beam inclination",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: Optional[str] = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(15, 5))
    plt.imshow(
        image,
        aspect="auto",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def save_return_visualizations(
    range_image_tensor: tf.Tensor,
    lidar_name: str,
    return_index: int,
    save_dir: Path,
    dpi: int,
    intensity_scale: float,
) -> None:
    array = range_image_tensor.numpy()
    ranges = array[..., 0]
    intensity = array[..., 1]
    elongation = array[..., 2]
    valid_mask = ranges > 0

    figures = [
        (
            log_range(ranges, valid_mask),
            "range_log",
            "Log-scaled range: log(1 + range)",
            None,
            None,
            None,
        ),
        (
            arctan_intensity(intensity, valid_mask, intensity_scale),
            "intensity_arctan",
            f"Arctangent-scaled intensity (scale={intensity_scale:g})",
            0.0,
            1.0,
            None,
        ),
        (elongation, "elongation", "Elongation", None, None, None),
        (
            valid_mask.astype(np.float32),
            "valid_mask",
            "Valid-return mask",
            0.0,
            1.0,
            "viridis",
        ),
    ]

    if array.shape[-1] > 3:
        figures.append(
            (array[..., 3], "channel_3", "Range-image channel 3", None, None, None)
        )

    return_number = return_index + 1
    for image, suffix, title, vmin, vmax, cmap in figures:
        save_image(
            image=image,
            title=f"{lidar_name} — return {return_number} — {title}",
            output_path=(
                save_dir
                / f"{lidar_name.lower()}_return_{return_number}_{suffix}.png"
            ),
            dpi=dpi,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )


def save_beam_diagnostics(
    lidar_name: str,
    beam_inclinations: tf.Tensor,
    save_dir: Path,
    dpi: int,
) -> None:
    beam_deg = np.rad2deg(beam_inclinations.numpy())
    rows = np.arange(beam_deg.size)

    plt.figure(figsize=(10, 5))
    plt.plot(rows, beam_deg, marker=".")
    plt.xlabel("Range-image row")
    plt.ylabel("Elevation angle [deg]")
    plt.title(f"{lidar_name} — row-to-beam inclination mapping")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = save_dir / f"{lidar_name.lower()}_beam_inclinations.png"
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    if beam_deg.size > 1:
        spacing = np.diff(beam_deg)
        plt.figure(figsize=(10, 5))
        plt.plot(rows[:-1], spacing, marker=".")
        plt.xlabel("Range-image row i")
        plt.ylabel("beam[i+1] - beam[i] [deg]")
        plt.title(f"{lidar_name} — adjacent beam-angle spacing")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        path = save_dir / f"{lidar_name.lower()}_beam_spacing.png"
        plt.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close()
        print(f"Saved: {path}")


def combine_return_images(
    return_1_tensor: tf.Tensor,
    return_2_tensor: tf.Tensor,
) -> Dict[str, np.ndarray]:
    r1 = return_1_tensor.numpy()
    r2 = return_2_tensor.numpy()
    if r1.shape != r2.shape:
        raise ValueError(f"Return image shapes differ: {r1.shape} vs {r2.shape}")

    range_1, range_2 = r1[..., 0], r2[..., 0]
    mask_1, mask_2 = range_1 > 0, range_2 > 0
    both = mask_1 & mask_2
    only_1 = mask_1 & ~mask_2
    only_2 = ~mask_1 & mask_2
    union = mask_1 | mask_2

    availability = np.zeros(range_1.shape, dtype=np.uint8)
    availability[only_1] = 1
    availability[only_2] = 2
    availability[both] = 3
    return_count = mask_1.astype(np.uint8) + mask_2.astype(np.uint8)

    nearest_range = np.zeros_like(range_1)
    farthest_range = np.zeros_like(range_1)
    nearest_range[only_1] = range_1[only_1]
    nearest_range[only_2] = range_2[only_2]
    nearest_range[both] = np.minimum(range_1[both], range_2[both])
    farthest_range[only_1] = range_1[only_1]
    farthest_range[only_2] = range_2[only_2]
    farthest_range[both] = np.maximum(range_1[both], range_2[both])

    range_gap = np.zeros_like(range_1)
    range_gap[both] = np.abs(range_2[both] - range_1[both])

    intensity_1, intensity_2 = r1[..., 1], r2[..., 1]
    elongation_1, elongation_2 = r1[..., 2], r2[..., 2]

    combined_intensity = np.zeros_like(intensity_1)
    combined_elongation = np.zeros_like(elongation_1)
    combined_intensity[only_1] = intensity_1[only_1]
    combined_intensity[only_2] = intensity_2[only_2]
    combined_intensity[both] = np.maximum(intensity_1[both], intensity_2[both])
    combined_elongation[only_1] = elongation_1[only_1]
    combined_elongation[only_2] = elongation_2[only_2]
    combined_elongation[both] = 0.5 * (
        elongation_1[both] + elongation_2[both]
    )

    return {
        "mask_return_1": mask_1,
        "mask_return_2": mask_2,
        "mask_both": both,
        "mask_only_return_1": only_1,
        "mask_only_return_2": only_2,
        "mask_union": union,
        "availability_code": availability,
        "return_count": return_count,
        "nearest_range": nearest_range,
        "farthest_range": farthest_range,
        "range_gap": range_gap,
        "combined_intensity": combined_intensity,
        "combined_elongation": combined_elongation,
        "range_difference_signed": range_2 - range_1,
        "intensity_difference_signed": intensity_2 - intensity_1,
        "elongation_difference_signed": elongation_2 - elongation_1,
    }


def combined_return_statistics(combined: Dict[str, np.ndarray]) -> Dict[str, Any]:
    total = combined["return_count"].size
    both = int(combined["mask_both"].sum())
    only_1 = int(combined["mask_only_return_1"].sum())
    only_2 = int(combined["mask_only_return_2"].sum())
    neither = int((combined["return_count"] == 0).sum())
    union = int(combined["mask_union"].sum())
    gap_values = combined["range_gap"][combined["mask_both"]]
    return {
        "total_pixels": total,
        "both_returns": both,
        "only_return_1": only_1,
        "only_return_2": only_2,
        "neither_return": neither,
        "union_valid_pixels": union,
        "union_valid_percentage": float(100.0 * union / total),
        "both_returns_percentage": float(100.0 * both / total),
        "range_gap_m": describe_values(gap_values),
    }


def save_combined_return_visualizations(
    lidar_name: str,
    combined: Dict[str, np.ndarray],
    save_dir: Path,
    dpi: int,
    intensity_scale: float,
) -> None:
    union = combined["mask_union"]
    save_image(
        combined["availability_code"],
        (
            f"{lidar_name} — return availability "
            "(0=none, 1=R1 only, 2=R2 only, 3=both)"
        ),
        save_dir / f"{lidar_name.lower()}_combined_return_availability.png",
        dpi,
        vmin=0,
        vmax=3,
        cmap="viridis",
    )
    save_image(
        combined["return_count"],
        f"{lidar_name} — number of valid returns per range-image pixel",
        save_dir / f"{lidar_name.lower()}_combined_return_count.png",
        dpi,
        vmin=0,
        vmax=2,
        cmap="viridis",
    )
    save_image(
        log_range(combined["nearest_range"], union),
        f"{lidar_name} — nearest valid return, log-scaled",
        save_dir / f"{lidar_name.lower()}_combined_nearest_range_log.png",
        dpi,
    )
    save_image(
        log_range(combined["farthest_range"], union),
        f"{lidar_name} — farthest valid return, log-scaled",
        save_dir / f"{lidar_name.lower()}_combined_farthest_range_log.png",
        dpi,
    )
    save_image(
        combined["range_gap"],
        f"{lidar_name} — |return 2 range - return 1 range| where both exist",
        save_dir / f"{lidar_name.lower()}_combined_range_gap.png",
        dpi,
    )
    save_image(
        arctan_intensity(combined["combined_intensity"], union, intensity_scale),
        f"{lidar_name} — combined intensity (max of valid returns), arctan-scaled",
        save_dir / f"{lidar_name.lower()}_combined_intensity_arctan.png",
        dpi,
        vmin=0,
        vmax=1,
    )
    save_image(
        combined["combined_elongation"],
        f"{lidar_name} — combined elongation (mean where both exist)",
        save_dir / f"{lidar_name.lower()}_combined_elongation.png",
        dpi,
    )


def robust_symmetric_limit(values: np.ndarray, mask: np.ndarray) -> float:
    selected = np.abs(values[mask & np.isfinite(values)])
    if selected.size == 0:
        return 1.0
    limit = float(np.percentile(selected, 99.0))
    return max(limit, 1e-6)


def save_comparison_panel(
    lidar_name: str,
    return_1_tensor: tf.Tensor,
    return_2_tensor: tf.Tensor,
    combined: Dict[str, np.ndarray],
    save_dir: Path,
    dpi: int,
    intensity_scale: float,
) -> None:
    r1, r2 = return_1_tensor.numpy(), return_2_tensor.numpy()
    m1, m2 = r1[..., 0] > 0, r2[..., 0] > 0
    both = combined["mask_both"]

    items = [
        (
            "range",
            log_range(r1[..., 0], m1),
            log_range(r2[..., 0], m2),
            combined["range_difference_signed"],
            "log(1+range)",
            "R2 - R1 range [m]",
            both,
        ),
        (
            "intensity",
            arctan_intensity(r1[..., 1], m1, intensity_scale),
            arctan_intensity(r2[..., 1], m2, intensity_scale),
            combined["intensity_difference_signed"],
            "arctan-scaled intensity",
            "R2 - R1 intensity",
            both,
        ),
        (
            "elongation",
            r1[..., 2],
            r2[..., 2],
            combined["elongation_difference_signed"],
            "elongation",
            "R2 - R1 elongation",
            both,
        ),
        (
            "validity",
            m1.astype(np.float32),
            m2.astype(np.float32),
            combined["return_count"].astype(np.float32),
            "valid mask",
            "valid-return count",
            np.ones_like(both, dtype=bool),
        ),
    ]

    for suffix, image_1, image_2, difference, value_label, diff_label, diff_mask in items:
        fig, axes = plt.subplots(3, 1, figsize=(15, 12), constrained_layout=True)
        im1 = axes[0].imshow(image_1, aspect="auto", interpolation="nearest")
        axes[0].set_title(f"{lidar_name} — return 1 — {value_label}")
        fig.colorbar(im1, ax=axes[0])

        im2 = axes[1].imshow(image_2, aspect="auto", interpolation="nearest")
        axes[1].set_title(f"{lidar_name} — return 2 — {value_label}")
        fig.colorbar(im2, ax=axes[1])

        if suffix == "validity":
            im3 = axes[2].imshow(
                difference,
                aspect="auto",
                interpolation="nearest",
                vmin=0,
                vmax=2,
            )
        else:
            display_diff = np.zeros_like(difference)
            display_diff[diff_mask] = difference[diff_mask]
            limit = robust_symmetric_limit(display_diff, diff_mask)
            im3 = axes[2].imshow(
                display_diff,
                aspect="auto",
                interpolation="nearest",
                cmap="coolwarm",
                vmin=-limit,
                vmax=limit,
            )
        axes[2].set_title(f"{lidar_name} — {diff_label}")
        fig.colorbar(im3, ax=axes[2])

        for axis in axes:
            axis.set_xlabel("Range-image column / azimuth sample")
            axis.set_ylabel("Range-image row / beam inclination")

        path = save_dir / f"{lidar_name.lower()}_return_comparison_{suffix}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")


def save_numpy_outputs(
    save_dir: Path,
    range_image_tensor: tf.Tensor,
    cartesian_range_image: tf.Tensor,
    points: np.ndarray,
    polar_features: np.ndarray,
    valid_indices: np.ndarray,
    beam_inclinations: tf.Tensor,
    extrinsic: np.ndarray,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / "range_image.npy", range_image_tensor.numpy())
    np.save(save_dir / "cartesian_range_image.npy", cartesian_range_image.numpy())
    np.save(save_dir / "point_cloud_xyz.npy", points)
    np.save(save_dir / "point_cloud_polar_features.npy", polar_features)
    np.save(save_dir / "valid_range_image_indices.npy", valid_indices)
    np.save(save_dir / "beam_inclinations_rad.npy", beam_inclinations.numpy())
    np.save(
        save_dir / "beam_inclinations_deg.npy",
        np.rad2deg(beam_inclinations.numpy()),
    )
    np.save(save_dir / "lidar_extrinsic.npy", extrinsic)


def save_combined_numpy_outputs(
    save_dir: Path,
    combined: Dict[str, np.ndarray],
    combined_points: np.ndarray,
    combined_polar_features: np.ndarray,
    combined_return_ids: np.ndarray,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    for name, array in combined.items():
        np.save(save_dir / f"{name}.npy", array)
    np.save(save_dir / "point_cloud_xyz_both_returns.npy", combined_points)
    np.save(
        save_dir / "point_cloud_polar_features_both_returns.npy",
        combined_polar_features,
    )
    np.save(save_dir / "point_cloud_return_ids.npy", combined_return_ids)


def write_ascii_ply(
    output_path: Path,
    points: np.ndarray,
    scalar: Optional[np.ndarray] = None,
    scalar_name: str = "return_id",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape [N,3], received {points.shape}")

    if scalar is not None:
        scalar = np.asarray(scalar).reshape(-1)
        if scalar.shape[0] != points.shape[0]:
            raise ValueError("PLY scalar length must equal number of points.")

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        if scalar is not None:
            handle.write(f"property int {scalar_name}\n")
        handle.write("end_header\n")
        if scalar is None:
            np.savetxt(handle, points, fmt="%.6f %.6f %.6f")
        else:
            values = np.column_stack([points, scalar.astype(np.int32)])
            np.savetxt(handle, values, fmt="%.6f %.6f %.6f %d")
    print(f"Saved: {output_path}")


def visualize_point_cloud_open3d(
    points: np.ndarray,
    return_ids: Optional[np.ndarray] = None,
    window_name: str = "Waymo point cloud",
) -> None:
    try:
        import open3d as o3d
    except ImportError:
        print("Open3D is not installed; skipping interactive visualization.", file=sys.stderr)
        return

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if return_ids is not None and len(return_ids) == len(points):
        ids = np.asarray(return_ids).reshape(-1)
        colors = np.zeros((len(ids), 3), dtype=np.float64)
        colors[ids == 1] = [0.15, 0.65, 0.95]
        colors[ids == 2] = [0.95, 0.45, 0.15]
        point_cloud.colors = o3d.utility.Vector3dVector(colors)

    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=2.0,
        origin=[0.0, 0.0, 0.0],
    )
    o3d.visualization.draw_geometries(
        [point_cloud, coordinate_frame],
        window_name=window_name,
    )


def extrinsic_metadata(extrinsic: np.ndarray) -> Dict[str, Any]:
    return {
        "matrix": extrinsic.tolist(),
        "rotation_matrix": extrinsic[:3, :3].tolist(),
        "translation_m": extrinsic[:3, 3].tolist(),
    }


def flatten_summary_rows(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for lidar_name, lidar_data in summary["lidars"].items():
        beam = lidar_data["beam_inclinations"]
        for return_name, return_data in lidar_data["returns"].items():
            row: Dict[str, Any] = {
                "lidar": lidar_name,
                "return": return_name,
                "height": return_data["height"],
                "width": return_data["width"],
                "channels": return_data["channels"],
                "total_pixels": return_data["total_pixels"],
                "valid_returns": return_data["valid_returns"],
                "valid_percentage": return_data["valid_percentage"],
                "range_min_m": return_data["range_m"]["min"],
                "range_mean_m": return_data["range_m"]["mean"],
                "range_max_m": return_data["range_m"]["max"],
                "intensity_median": return_data["intensity"]["median"],
                "intensity_p995": return_data["intensity"]["p995"],
                "intensity_max": return_data["intensity"]["max"],
                "elongation_mean": return_data["elongation"]["mean"],
                "beam_source": beam["source"],
                "beam_count": beam["beam_count"],
                "elevation_max_deg": beam["maximum_elevation_deg"],
                "elevation_min_deg": beam["minimum_elevation_deg"],
                "vertical_fov_deg": beam["vertical_fov_deg"],
                "extrinsic_x_m": lidar_data["extrinsic"]["translation_m"][0],
                "extrinsic_y_m": lidar_data["extrinsic"]["translation_m"][1],
                "extrinsic_z_m": lidar_data["extrinsic"]["translation_m"][2],
            }
            rows.append(row)
    return rows


def save_summary_files(summary: Dict[str, Any], save_dir: Path) -> None:
    json_path = save_dir / "waymo_lidar_debug_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved: {json_path}")

    rows = flatten_summary_rows(summary)
    if rows:
        csv_path = save_dir / "waymo_lidar_debug_summary.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved: {csv_path}")

    text_lines: List[str] = []
    text_lines.append("WAYMO LIDAR RANGE-IMAGE DEBUG SUMMARY")
    text_lines.append("=" * 80)
    frame_info = summary["frame"]
    text_lines.extend(
        [
            f"TFRecord: {frame_info['tfrecord']}",
            f"Frame index: {frame_info['frame_index']}",
            f"Timestamp [us]: {frame_info['timestamp_micros']}",
            "",
        ]
    )

    for lidar_name, data in summary["lidars"].items():
        beam = data["beam_inclinations"]
        translation = data["extrinsic"]["translation_m"]
        text_lines.append(lidar_name)
        text_lines.append("-" * len(lidar_name))
        text_lines.append(
            "Range-image structure: "
            + ", ".join(
                f"{name}={value['height']}x{value['width']}x{value['channels']}"
                for name, value in data["returns"].items()
            )
        )
        text_lines.append(
            f"Beam source: {beam['source']}; beam count: {beam['beam_count']}"
        )
        text_lines.append(
            f"Elevation: {beam['maximum_elevation_deg']:.6f} deg to "
            f"{beam['minimum_elevation_deg']:.6f} deg; "
            f"vertical FOV={beam['vertical_fov_deg']:.6f} deg"
        )
        text_lines.append(
            "Extrinsic translation [m]: "
            f"x={translation[0]:.6f}, y={translation[1]:.6f}, z={translation[2]:.6f}"
        )
        for return_name, return_data in data["returns"].items():
            text_lines.append(
                f"  {return_name}: valid={return_data['valid_returns']}/"
                f"{return_data['total_pixels']} "
                f"({return_data['valid_percentage']:.2f}%), "
                f"range={return_data['range_m']['min']:.3f}.."
                f"{return_data['range_m']['max']:.3f} m, "
                f"mean={return_data['range_m']['mean']:.3f} m"
            )
        if data.get("combined_returns"):
            combined = data["combined_returns"]
            text_lines.append(
                "  combined: "
                f"both={combined['both_returns']}, "
                f"R1-only={combined['only_return_1']}, "
                f"R2-only={combined['only_return_2']}, "
                f"neither={combined['neither_return']}, "
                f"union={combined['union_valid_pixels']} "
                f"({combined['union_valid_percentage']:.2f}%)"
            )
        text_lines.append("")

    text_path = save_dir / "waymo_lidar_debug_summary.txt"
    text_path.write_text("\n".join(text_lines), encoding="utf-8")
    print(f"Saved: {text_path}")


def main() -> None:
    args = parse_arguments()
    args.save_dir.mkdir(parents=True, exist_ok=True)

    frame = load_frame(args.tfrecord, args.frame_index)
    print_available_lidars(frame)
    range_images, range_image_top_pose = parse_range_images(frame)

    summary: Dict[str, Any] = {
        "frame": {
            "tfrecord": str(args.tfrecord),
            "frame_index": args.frame_index,
            "timestamp_micros": int(frame.timestamp_micros),
            "number_lidars": len(frame.lasers),
            "number_images": len(frame.images),
            "frame_pose": np.asarray(frame.pose.transform, dtype=np.float32)
            .reshape(4, 4)
            .tolist(),
            "top_pixel_pose_statistics": top_pixel_pose_statistics(
                range_image_top_pose
            ),
        },
        "lidars": {},
    }

    frame_all_points: List[np.ndarray] = []
    frame_all_sensor_ids: List[np.ndarray] = []
    frame_all_return_ids: List[np.ndarray] = []

    for lidar_name in requested_lidars(args.lidar):
        lidar_enum = lidar_name_to_enum(lidar_name)
        if lidar_enum not in range_images or not range_images[lidar_enum]:
            print(f"No range images found for {lidar_name}; skipping.")
            continue

        print("\n" + "=" * 88)
        print(f"PROCESSING LIDAR: {lidar_name}")
        print("=" * 88)

        lidar_dir = args.save_dir / lidar_name.lower()
        lidar_dir.mkdir(parents=True, exist_ok=True)

        calibration = find_laser_calibration(frame, lidar_enum)
        lidar_summary: Dict[str, Any] = {
            "enum": int(lidar_enum),
            "returns": {},
        }
        processed: Dict[int, Dict[str, Any]] = {}

        for return_index in requested_return_indices(args.returns):
            matrix = range_images[lidar_enum].get(return_index)
            if matrix is None:
                print(f"{lidar_name} return {return_index + 1} is unavailable; skipping.")
                continue

            return_name = RETURN_LABELS[return_index]
            return_dir = lidar_dir / return_name
            return_dir.mkdir(parents=True, exist_ok=True)

            tensor = matrix_float_to_tensor(matrix)
            stats = range_image_statistics(tensor, lidar_name, return_index)
            cartesian, beams, extrinsic, beam_metadata = (
                convert_range_image_to_cartesian(
                    frame,
                    lidar_enum,
                    tensor,
                    range_image_top_pose,
                )
            )
            points, polar_features, valid_indices = form_point_cloud(
                tensor,
                cartesian,
            )

            print("\nCartesian range image")
            print("---------------------")
            print(f"Shape:                {tuple(cartesian.shape)}")
            print("Meaning: each pixel now stores [x, y, z]")
            print("\nPoint cloud")
            print("-----------")
            print(f"XYZ shape:            {points.shape}")
            print(f"Polar features shape: {polar_features.shape}")
            print(f"Valid-index shape:    {valid_indices.shape}")

            save_numpy_outputs(
                return_dir,
                tensor,
                cartesian,
                points,
                polar_features,
                valid_indices,
                beams,
                extrinsic,
            )
            save_return_visualizations(
                tensor,
                lidar_name,
                return_index,
                return_dir,
                args.dpi,
                args.intensity_scale,
            )
            if args.save_ply:
                write_ascii_ply(return_dir / "point_cloud_xyz.ply", points)

            processed[return_index] = {
                "tensor": tensor,
                "cartesian": cartesian,
                "points": points,
                "polar_features": polar_features,
                "valid_indices": valid_indices,
                "beams": beams,
                "extrinsic": extrinsic,
                "beam_metadata": beam_metadata,
            }
            lidar_summary["returns"][return_name] = stats

            frame_all_points.append(points)
            frame_all_sensor_ids.append(
                np.full(len(points), lidar_enum, dtype=np.int32)
            )
            frame_all_return_ids.append(
                np.full(len(points), return_index + 1, dtype=np.int32)
            )

        if not processed:
            continue

        # Beam and calibration are identical for both returns; save once per LiDAR.
        reference = processed[min(processed.keys())]
        lidar_summary["beam_inclinations"] = reference["beam_metadata"]
        lidar_summary["extrinsic"] = extrinsic_metadata(reference["extrinsic"])
        save_beam_diagnostics(
            lidar_name,
            reference["beams"],
            lidar_dir,
            args.dpi,
        )
        np.save(lidar_dir / "lidar_extrinsic.npy", reference["extrinsic"])
        np.save(
            lidar_dir / "beam_inclinations_deg.npy",
            np.rad2deg(reference["beams"].numpy()),
        )

        # Only meaningful when both returns are available and requested.
        if 0 in processed and 1 in processed:
            combined_dir = lidar_dir / "combined_returns"
            combined = combine_return_images(
                processed[0]["tensor"],
                processed[1]["tensor"],
            )
            combined_stats = combined_return_statistics(combined)
            lidar_summary["combined_returns"] = combined_stats

            print("\nCombined-return statistics")
            print("--------------------------")
            print(f"Both returns:         {combined_stats['both_returns']}")
            print(f"Return 1 only:        {combined_stats['only_return_1']}")
            print(f"Return 2 only:        {combined_stats['only_return_2']}")
            print(f"Neither return:       {combined_stats['neither_return']}")
            print(f"Union valid pixels:   {combined_stats['union_valid_pixels']}")
            print(
                f"Union valid percent:  "
                f"{combined_stats['union_valid_percentage']:.2f}%"
            )

            combined_points = np.concatenate(
                [processed[0]["points"], processed[1]["points"]], axis=0
            )
            combined_polar = np.concatenate(
                [
                    processed[0]["polar_features"],
                    processed[1]["polar_features"],
                ],
                axis=0,
            )
            combined_return_ids = np.concatenate(
                [
                    np.ones(len(processed[0]["points"]), dtype=np.int32),
                    np.full(len(processed[1]["points"]), 2, dtype=np.int32),
                ]
            )

            save_combined_numpy_outputs(
                combined_dir,
                combined,
                combined_points,
                combined_polar,
                combined_return_ids,
            )
            save_combined_return_visualizations(
                lidar_name,
                combined,
                combined_dir,
                args.dpi,
                args.intensity_scale,
            )
            if args.save_comparison_panels:
                save_comparison_panel(
                    lidar_name,
                    processed[0]["tensor"],
                    processed[1]["tensor"],
                    combined,
                    combined_dir,
                    args.dpi,
                    args.intensity_scale,
                )
            if args.save_ply:
                write_ascii_ply(
                    combined_dir / "point_cloud_both_returns.ply",
                    combined_points,
                    combined_return_ids,
                    scalar_name="return_id",
                )

            if args.visualize_open3d and args.lidar != "ALL":
                visualize_point_cloud_open3d(
                    combined_points,
                    combined_return_ids,
                    window_name=f"{lidar_name}: return 1 + return 2",
                )
        elif args.returns == "BOTH":
            print(
                f"Cannot create combined-return outputs for {lidar_name}: "
                "both returns were not available."
            )

        summary["lidars"][lidar_name] = lidar_summary

    # Frame-level point cloud: all processed sensors and returns.
    if frame_all_points:
        frame_dir = args.save_dir / "all_lidars_combined"
        frame_dir.mkdir(parents=True, exist_ok=True)
        all_points = np.concatenate(frame_all_points, axis=0)
        all_sensor_ids = np.concatenate(frame_all_sensor_ids, axis=0)
        all_return_ids = np.concatenate(frame_all_return_ids, axis=0)
        np.save(frame_dir / "point_cloud_xyz_all_lidars_all_returns.npy", all_points)
        np.save(frame_dir / "sensor_ids.npy", all_sensor_ids)
        np.save(frame_dir / "return_ids.npy", all_return_ids)
        if args.save_ply:
            # Pack sensor and return into one integer for a simple PLY scalar:
            # 10 * sensor_enum + return_id, e.g. TOP/R1 -> 11, SIDE_LEFT/R2 -> 32.
            labels = 10 * all_sensor_ids + all_return_ids
            write_ascii_ply(
                frame_dir / "point_cloud_all_lidars_all_returns.ply",
                all_points,
                labels,
                scalar_name="sensor_return_code",
            )
        summary["frame"]["all_processed_point_count"] = int(len(all_points))

    save_summary_files(summary, args.save_dir)
    print("\nDone.")
    print(f"All outputs are under: {args.save_dir.resolve()}")


if __name__ == "__main__":
    main()