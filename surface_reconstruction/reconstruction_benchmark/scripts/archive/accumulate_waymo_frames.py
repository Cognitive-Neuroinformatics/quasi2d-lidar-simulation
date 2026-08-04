#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml

from waymo_open_dataset import dataset_pb2

from scripts.waymo_frame_utils import (
    crop_xyz,
    extract_lidar_points,
    get_vehicle_pose,
    iterate_waymo_frames,
    transform_xyz,
)


def load_yaml(path: Path) -> dict:
    """Load a YAML configuration file."""

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML configuration: {path}")

    return config


def save_point_cloud(
    points: np.ndarray,
    output_path: Path,
) -> None:
    """Save an Nx3 point array as a PLY point cloud."""

    points = np.asarray(points, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"Expected points with shape (N, 3), got {points.shape}"
        )

    if len(points) == 0:
        raise ValueError(
            f"Cannot save an empty point cloud: {output_path}"
        )

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    success = o3d.io.write_point_cloud(
        str(output_path),
        cloud,
        write_ascii=False,
        compressed=False,
    )

    if not success:
        raise RuntimeError(
            f"Failed to save point cloud: {output_path}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Accumulate multiple Waymo LiDAR frames in the "
            "coordinate system of a reference frame."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML configuration file.",
    )

    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="First frame index to include.",
    )

    parser.add_argument(
        "--num-frames",
        type=int,
        default=10,
        help="Number of consecutive frames to accumulate.",
    )

    parser.add_argument(
        "--reference-frame",
        type=int,
        default=None,
        help=(
            "Frame used as the output coordinate system. "
            "Defaults to --start-frame."
        ),
    )

    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.05,
        help="Voxel size in metres for the downsampled output.",
    )

    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path(
            "data/accumulated/frames_000_009_top"
        ),
        help=(
            "Output path prefix. The script adds suffixes for "
            "raw PLY, voxelized PLY, NPZ and metadata JSON."
        ),
    )

    args = parser.parse_args()

    if args.start_frame < 0:
        raise ValueError(
            "--start-frame must be zero or greater"
        )

    if args.num_frames <= 0:
        raise ValueError(
            "--num-frames must be positive"
        )

    if args.voxel_size <= 0:
        raise ValueError(
            "--voxel-size must be positive"
        )

    reference_index = (
        args.start_frame
        if args.reference_frame is None
        else args.reference_frame
    )

    end_frame_exclusive = (
        args.start_frame + args.num_frames
    )

    if not (
        args.start_frame
        <= reference_index
        < end_frame_exclusive
    ):
        raise ValueError(
            "The reference frame must lie inside the "
            "accumulation interval"
        )

    config = load_yaml(args.config)

    if "input" not in config:
        raise KeyError(
            "Missing 'input' section in the YAML configuration"
        )

    input_config = config["input"]

    if "tfrecord" not in input_config:
        raise KeyError(
            "Missing 'input.tfrecord' in the YAML configuration"
        )

    tfrecord_path = Path(
        input_config["tfrecord"]
    ).expanduser().resolve()

    if not tfrecord_path.exists():
        raise FileNotFoundError(
            f"TFRecord not found: {tfrecord_path}"
        )

    lidar_names = input_config.get(
        "lidar_names",
        ["TOP"],
    )

    if not isinstance(lidar_names, list) or not lidar_names:
        raise ValueError(
            "input.lidar_names must be a non-empty list"
        )

    lidar_names = [
        str(name).strip().upper()
        for name in lidar_names
    ]

    return_index = int(
        input_config.get("return_index", 0)
    )

    if return_index not in (0, 1):
        raise ValueError(
            "input.return_index must be 0 or 1"
        )

    return_number = return_index + 1

    crop_config = config.get(
        "crop",
        {"enabled": False},
    )

    selected_frames: list[
        tuple[int, dataset_pb2.Frame]
    ] = []

    for frame_index, frame in iterate_waymo_frames(
        tfrecord_path
    ):
        if frame_index < args.start_frame:
            continue

        if frame_index >= end_frame_exclusive:
            break

        selected_frames.append(
            (frame_index, frame)
        )

    if len(selected_frames) != args.num_frames:
        raise RuntimeError(
            f"Requested {args.num_frames} frames starting at "
            f"{args.start_frame}, but only found "
            f"{len(selected_frames)}"
        )

    reference_frame = next(
        frame
        for frame_index, frame in selected_frames
        if frame_index == reference_index
    )

    reference_pose = get_vehicle_pose(
        reference_frame
    )

    if reference_pose.shape != (4, 4):
        raise ValueError(
            f"Unexpected reference pose shape: "
            f"{reference_pose.shape}"
        )

    world_to_reference = np.linalg.inv(
        reference_pose
    )

    accumulated_parts: list[np.ndarray] = []
    frame_statistics: list[dict] = []

    print(f"TFRecord:        {tfrecord_path}")
    print(
        f"Frame interval:  "
        f"{args.start_frame}–"
        f"{end_frame_exclusive - 1}"
    )
    print(f"Reference frame: {reference_index}")
    print(f"LiDARs:          {lidar_names}")
    print(f"Return index:    {return_index}")
    print(f"Return number:   {return_number}")
    print(f"Voxel size:      {args.voxel_size:.3f} m")

    for frame_index, frame in selected_frames:
        (
            points_vehicle,
            _intensity,
            _elongation,
            _sensor_origins,
        ) = extract_lidar_points(
            frame=frame,
            requested_lidars=lidar_names,
            return_index=return_index,
        )

        points_vehicle = np.asarray(
            points_vehicle,
            dtype=np.float64,
        )

        if (
            points_vehicle.ndim != 2
            or points_vehicle.shape[1] != 3
        ):
            raise ValueError(
                f"Unexpected point shape for frame "
                f"{frame_index}: {points_vehicle.shape}"
            )

        finite_mask = np.isfinite(
            points_vehicle
        ).all(axis=1)

        points_vehicle = points_vehicle[
            finite_mask
        ]

        frame_pose = get_vehicle_pose(frame)

        if frame_pose.shape != (4, 4):
            raise ValueError(
                f"Unexpected pose shape for frame "
                f"{frame_index}: {frame_pose.shape}"
            )

        vehicle_to_reference = (
            world_to_reference @ frame_pose
        )

        points_reference = transform_xyz(
            points_vehicle,
            vehicle_to_reference,
        )

        (
            points_reference,
            _crop_mask,
        ) = crop_xyz(
            points_reference,
            crop_config,
        )

        if len(points_reference) == 0:
            print(
                f"Warning: frame {frame_index:03d} "
                "contains no points after cropping"
            )
            continue

        accumulated_parts.append(
            points_reference
        )

        translation = (
            vehicle_to_reference[:3, 3]
        )

        frame_statistics.append(
            {
                "frame_index": int(frame_index),
                "timestamp_micros": int(
                    frame.timestamp_micros
                ),
                "raw_lidar_points": int(
                    len(points_vehicle)
                ),
                "cropped_points": int(
                    len(points_reference)
                ),
                "translation_to_reference": (
                    translation.tolist()
                ),
                "vehicle_to_reference": (
                    vehicle_to_reference.tolist()
                ),
            }
        )

        print(
            f"Frame {frame_index:03d}: "
            f"{len(points_vehicle):,} raw → "
            f"{len(points_reference):,} cropped, "
            f"translation="
            f"[{translation[0]:.3f}, "
            f"{translation[1]:.3f}, "
            f"{translation[2]:.3f}]"
        )

    if not accumulated_parts:
        raise RuntimeError(
            "No points remain after extraction and cropping"
        )

    accumulated = np.concatenate(
        accumulated_parts,
        axis=0,
    )

    finite_accumulated_mask = np.isfinite(
        accumulated
    ).all(axis=1)

    accumulated = accumulated[
        finite_accumulated_mask
    ]

    if len(accumulated) == 0:
        raise RuntimeError(
            "Accumulated point cloud contains no valid points"
        )

    print(
        f"\nAccumulated before downsampling: "
        f"{len(accumulated):,}"
    )

    raw_output = args.output_prefix.with_name(
        args.output_prefix.name + "_raw.ply"
    )

    voxel_output = args.output_prefix.with_name(
        args.output_prefix.name
        + f"_voxel_{args.voxel_size:.3f}.ply"
    )

    npz_output = args.output_prefix.with_suffix(
        ".npz"
    )

    metadata_output = (
        args.output_prefix.with_name(
            args.output_prefix.name
            + "_metadata.json"
        )
    )

    save_point_cloud(
        accumulated,
        raw_output,
    )

    raw_cloud = o3d.geometry.PointCloud()
    raw_cloud.points = (
        o3d.utility.Vector3dVector(
            accumulated
        )
    )

    voxel_cloud = raw_cloud.voxel_down_sample(
        args.voxel_size
    )

    voxel_points = np.asarray(
        voxel_cloud.points,
        dtype=np.float32,
    )

    if len(voxel_points) == 0:
        raise RuntimeError(
            "Voxel downsampling produced an empty cloud"
        )

    save_point_cloud(
        voxel_points,
        voxel_output,
    )

    npz_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        npz_output,
        xyz=voxel_points,
        reference_pose=reference_pose,
        world_to_reference=world_to_reference,
    )

    metadata = {
        "source_tfrecord": str(
            tfrecord_path
        ),
        "start_frame": int(
            args.start_frame
        ),
        "end_frame_inclusive": int(
            end_frame_exclusive - 1
        ),
        "num_frames_requested": int(
            args.num_frames
        ),
        "num_frames_accumulated": int(
            len(frame_statistics)
        ),
        "reference_frame": int(
            reference_index
        ),
        "lidar_names": lidar_names,
        "return_index": int(
            return_index
        ),
        "return_number": int(
            return_number
        ),
        "coordinate_frame": (
            f"vehicle_frame_{reference_index}"
        ),
        "dynamic_object_removal": False,
        "crop": crop_config,
        "voxel_size": float(
            args.voxel_size
        ),
        "raw_accumulated_points": int(
            len(accumulated)
        ),
        "voxelized_points": int(
            len(voxel_points)
        ),
        "reference_pose_vehicle_to_world": (
            reference_pose.tolist()
        ),
        "frames": frame_statistics,
    }

    metadata_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with metadata_output.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    print("\nSaved")
    print("-----")
    print(
        f"Raw accumulation: {raw_output}"
    )
    print(
        f"Voxelized cloud:  {voxel_output}"
    )
    print(
        f"NPZ:              {npz_output}"
    )
    print(
        f"Metadata:         {metadata_output}"
    )
    print(
        f"Voxelized points: {len(voxel_points):,}"
    )


if __name__ == "__main__":
    main()