#!/usr/bin/env python3
"""
Dynamic Object Extraction and Static Scene Accumulation
======================================================

This script prepares a dynamic Waymo scene for surface reconstruction by
separating the environment into:

1. A dense static scene.
2. Canonical geometry for every Vehicle, Pedestrian and Cyclist.
3. Per-frame object poses for later scene composition.

The implementation follows the idea that static geometry can be accumulated
across multiple frames into a common coordinate system, while moving objects
must be reconstructed separately to avoid motion blur and ghosting.

Pipeline

For every selected Waymo frame:

1. Extract LiDAR points from the sensors (currently only TOP waymo LIDAR is being checked with return 0).
2. Obtain all annotated object bounding boxes
3. Transform points inside each bounding box into a canonical object
   coordinate system whose:
       - origin is the box center,
       - x-axis points forward,
       - y-axis points left,
       - z-axis points upward.
4. Store each object crop together with all coordinate transforms
   required to move between:
       object frame
       vehicle frame
       reference frame
5. Remove the object points from the scene.
6. Transform the remaining static points into the reference frame.
7. Accumulate static geometry from all frames.

After all frames have been processed:

• Merge all canonical observations belonging to the same track ID to
  obtain a denser canonical representation of each dynamic object.

• Merge all static points into a dense static point cloud.

• Apply voxel downsampling to both the static scene and every canonical
  object.

Outputs

The script produces:

- Dense accumulated static point cloud.
- Voxelized static point cloud.
- Canonical point cloud for every tracked dynamic object.
- Per-frame object observations.
- Complete object trajectories and rigid transforms.
- Metadata describing every frame and every object.

Coordinate Systems

Vehicle frame:
    Native Waymo coordinate system for each frame (baselink).

Reference frame:
    Vehicle coordinate system of the selected reference frame (currently frame 0).
    All static geometry is accumulated here.

Canonical object frame:
    Local object coordinate system centered at the object's bounding box.
    All observations of the same object are expressed in this frame before
    being merged.

Further:

Accumulating the complete scene directly produces duplicated moving vehicles
and blurred geometry. By reconstructing static and dynamic components
independently, a dense and temporally consistent scene can later be composed
for any frame by placing each canonical object back into its recorded pose.

This representation serves as the input for subsequent surface
reconstruction methods such as BPA, GP3, Screened Poisson and LiDAR-GS.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
import yaml

from waymo_open_dataset import dataset_pb2
from waymo_open_dataset import label_pb2

from scripts.waymo_frame_utils import (
    crop_xyz,
    extract_lidar_points,
    get_vehicle_pose,
    iterate_waymo_frames,
    transform_xyz,
)


FOREGROUND_TYPES = {
    label_pb2.Label.Type.TYPE_VEHICLE,
    label_pb2.Label.Type.TYPE_PEDESTRIAN,
    label_pb2.Label.Type.TYPE_CYCLIST,
}

def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML configuration: {path}")

    return config


def save_point_cloud(
    xyz: np.ndarray,
    path: Path,
) -> None:
    xyz = np.asarray(xyz, dtype=np.float64)

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(
            f"Expected point array with shape (N, 3), got {xyz.shape}"
        )

    if len(xyz) == 0:
        return

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(xyz)

    path.parent.mkdir(parents=True, exist_ok=True)

    if not o3d.io.write_point_cloud(
        str(path),
        cloud,
        write_ascii=False,
        compressed=False,
    ):
        raise RuntimeError(f"Failed to save point cloud: {path}")


def label_type_name(label_type: int) -> str:
    mapping = {
        label_pb2.Label.Type.TYPE_UNKNOWN: "UNKNOWN",
        label_pb2.Label.Type.TYPE_VEHICLE: "VEHICLE",
        label_pb2.Label.Type.TYPE_PEDESTRIAN: "PEDESTRIAN",
        label_pb2.Label.Type.TYPE_SIGN: "SIGN",
        label_pb2.Label.Type.TYPE_CYCLIST: "CYCLIST",
    }

    return mapping.get(
        label_type,
        f"TYPE_{label_type}",
    )


def rotation_z(heading: float) -> np.ndarray:
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)

    return np.array(
        [
            [cos_h, -sin_h, 0.0],
            [sin_h, cos_h, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def vehicle_to_object_coordinates(
    points_vehicle: np.ndarray,
    center_vehicle: np.ndarray,
    heading: float,
) -> np.ndarray:
    """
    Convert vehicle-frame points to a canonical object frame.

    Object-frame convention:
        x: object forward
        y: object left
        z: upward
        origin: 3D box center
    """

    rotation_object_to_vehicle = rotation_z(heading)

    relative = points_vehicle - center_vehicle[None, :]

    # Row-vector equivalent of:
    # p_object = R_object_to_vehicle.T @ (p_vehicle - center)
    return relative @ rotation_object_to_vehicle


def points_inside_box(
    points_vehicle: np.ndarray,
    center_vehicle: np.ndarray,
    heading: float,
    length: float,
    width: float,
    height: float,
    margin_xy: float,
    margin_z: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return:
        inside_mask
        canonical coordinates for all input points
    """

    points_object = vehicle_to_object_coordinates(
        points_vehicle=points_vehicle,
        center_vehicle=center_vehicle,
        heading=heading,
    )

    half_length = 0.5 * length + margin_xy
    half_width = 0.5 * width + margin_xy
    half_height = 0.5 * height + margin_z

    inside_mask = (
        (np.abs(points_object[:, 0]) <= half_length)
        & (np.abs(points_object[:, 1]) <= half_width)
        & (np.abs(points_object[:, 2]) <= half_height)
    )

    return inside_mask, points_object


def make_object_to_vehicle_transform(
    center_vehicle: np.ndarray,
    heading: float,
) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_z(heading)
    transform[:3, 3] = center_vehicle
    return transform


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Remove Waymo foreground objects from static accumulation "
            "and save per-track canonical object crops."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--reference-frame",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--box-margin-xy",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--box-margin-z",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--minimum-object-points",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "data/dynamic_objects/frames_000_009"
        ),
    )

    args = parser.parse_args()

    if args.start_frame < 0:
        raise ValueError("--start-frame must be non-negative")

    if args.num_frames <= 0:
        raise ValueError("--num-frames must be positive")

    if args.voxel_size <= 0:
        raise ValueError("--voxel-size must be positive")

    if args.box_margin_xy < 0 or args.box_margin_z < 0:
        raise ValueError("Box margins must be non-negative")

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
            "Reference frame must lie inside the accumulation interval"
        )

    config = load_yaml(args.config)
    input_config = config["input"]

    tfrecord_path = Path(
        input_config["tfrecord"]
    ).expanduser().resolve()

    lidar_names = [
        str(name).strip().upper()
        for name in input_config.get(
            "lidar_names",
            ["TOP"],
        )
    ]

    return_index = int(
        input_config.get("return_index", 0)
    )

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
            f"Requested {args.num_frames} frames, "
            f"but found {len(selected_frames)}"
        )

    reference_frame = next(
        frame
        for frame_index, frame in selected_frames
        if frame_index == reference_index
    )

    reference_pose = get_vehicle_pose(
        reference_frame
    )

    world_to_reference = np.linalg.inv(
        reference_pose
    )

    static_parts_reference: list[np.ndarray] = []

    canonical_track_parts: dict[
        str,
        list[np.ndarray],
    ] = defaultdict(list)

    track_metadata: dict[
        str,
        dict[str, Any],
    ] = {}

    frame_summary: list[dict[str, Any]] = []

    output_root = args.output_root
    tracks_root = output_root / "tracks"
    per_frame_root = output_root / "per_frame"

    tracks_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    per_frame_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    static_per_frame_directory = (
        output_root / "static_per_frame"
    )

    static_per_frame_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"TFRecord:        {tfrecord_path}")
    print(
        f"Frames:          "
        f"{args.start_frame}–{end_frame_exclusive - 1}"
    )
    print(f"Reference frame: {reference_index}")
    print(f"LiDARs:          {lidar_names}")
    print(f"Return index:    {return_index}")

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

        finite_mask = np.isfinite(
            points_vehicle
        ).all(axis=1)

        points_vehicle = points_vehicle[
            finite_mask
        ]

        foreground_mask = np.zeros(
            len(points_vehicle),
            dtype=bool,
        )

        frame_objects: list[dict[str, Any]] = []

        frame_pose = get_vehicle_pose(frame)

        vehicle_to_reference = (
            world_to_reference @ frame_pose
        )


        for label in frame.laser_labels:
            if label.type not in FOREGROUND_TYPES:
                continue

            box = label.box

            center_vehicle = np.array(
                [
                    box.center_x,
                    box.center_y,
                    box.center_z,
                ],
                dtype=np.float64,
            )

            inside_mask, points_object_all = points_inside_box(
                points_vehicle=points_vehicle,
                center_vehicle=center_vehicle,
                heading=float(box.heading),
                length=float(box.length),
                width=float(box.width),
                height=float(box.height),
                margin_xy=args.box_margin_xy,
                margin_z=args.box_margin_z,
            )

            foreground_mask |= inside_mask

            object_points = points_object_all[
                inside_mask
            ]

            if len(object_points) < args.minimum_object_points:
                continue

            track_id = str(label.id)

            object_to_vehicle = (
                make_object_to_vehicle_transform(
                    center_vehicle=center_vehicle,
                    heading=float(box.heading),
                )
            )

            vehicle_to_object = np.linalg.inv(
                object_to_vehicle
            )

            object_to_reference = (
                vehicle_to_reference
                @ object_to_vehicle
            )

            reference_to_object = np.linalg.inv(
                object_to_reference
            )

            frame_track_directory = (
                per_frame_root
                / f"frame_{frame_index:03d}"
            )

            frame_track_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            npz_path = (
                frame_track_directory
                / f"{track_id}.npz"
            )

            np.savez_compressed(
                npz_path,
                xyz_object=object_points.astype(
                    np.float32
                ),
                center_vehicle=center_vehicle,
                heading=np.float64(box.heading),
                length=np.float64(box.length),
                width=np.float64(box.width),
                height=np.float64(box.height),
                object_to_vehicle=object_to_vehicle,
                vehicle_to_object=vehicle_to_object,
                vehicle_to_reference=vehicle_to_reference,
                object_to_reference=object_to_reference,
                reference_to_object=reference_to_object,
                frame_index=np.int32(frame_index),
                timestamp_micros=np.int64(
                    frame.timestamp_micros
                ),
            )

            canonical_track_parts[
                track_id
            ].append(object_points)

            object_record = {
                "frame_index": int(frame_index),
                "timestamp_micros": int(
                    frame.timestamp_micros
                ),
                "track_id": track_id,
                "type": label_type_name(
                    label.type
                ),
                "num_points": int(
                    len(object_points)
                ),
                "center_vehicle": (
                    center_vehicle.tolist()
                ),
                "heading": float(
                    box.heading
                ),
                "length": float(
                    box.length
                ),
                "width": float(
                    box.width
                ),
                "height": float(
                    box.height
                ),
                "object_to_vehicle": (
                    object_to_vehicle.tolist()
                ),
                "vehicle_to_object": (
                    vehicle_to_object.tolist()
                ),
                "crop_file": str(npz_path),
                "vehicle_to_reference": (
                    vehicle_to_reference.tolist()
                ),
                "object_to_reference": (
                    object_to_reference.tolist()
                ),
                "reference_to_object": (
                    reference_to_object.tolist()
                ),
            }

            frame_objects.append(
                object_record
            )

            if track_id not in track_metadata:
                track_metadata[track_id] = {
                    "track_id": track_id,
                    "type": label_type_name(
                        label.type
                    ),
                    "observations": [],
                }

            track_metadata[
                track_id
            ]["observations"].append(
                object_record
            )

        static_points_vehicle = points_vehicle[
            ~foreground_mask
        ]

        save_point_cloud(
            static_points_vehicle,
            static_per_frame_directory
            / f"frame_{frame_index:03d}.ply",
        )

        static_points_reference = transform_xyz(
            static_points_vehicle,
            vehicle_to_reference,
        )

        (
            static_points_reference,
            _crop_mask,
        ) = crop_xyz(
            static_points_reference,
            crop_config,
        )

        static_parts_reference.append(
            static_points_reference
        )

        frame_summary.append(
            {
                "frame_index": int(frame_index),
                "timestamp_micros": int(
                    frame.timestamp_micros
                ),
                "vehicle_to_reference": (
                    vehicle_to_reference.tolist()
                ),
                "raw_points": int(
                    len(points_vehicle)
                ),
                "removed_foreground_points": int(
                    foreground_mask.sum()
                ),
                "static_points_after_crop": int(
                    len(static_points_reference)
                ),
                "num_saved_objects": int(
                    len(frame_objects)
                ),
                "objects": frame_objects,
            }
        )

        print(
            f"Frame {frame_index:03d}: "
            f"{len(points_vehicle):,} raw, "
            f"{foreground_mask.sum():,} foreground removed, "
            f"{len(static_points_reference):,} static, "
            f"{len(frame_objects)} object crops"
        )

    static_raw = np.concatenate(
        static_parts_reference,
        axis=0,
    )

    static_cloud = o3d.geometry.PointCloud()
    static_cloud.points = (
        o3d.utility.Vector3dVector(
            static_raw
        )
    )

    static_voxel_cloud = (
        static_cloud.voxel_down_sample(
            args.voxel_size
        )
    )

    static_voxel = np.asarray(
        static_voxel_cloud.points,
        dtype=np.float32,
    )

    save_point_cloud(
        static_raw,
        output_root / "static_raw.ply",
    )

    save_point_cloud(
        static_voxel,
        output_root
        / f"static_voxel_{args.voxel_size:.3f}.ply",
    )

    for track_id, parts in canonical_track_parts.items():
        canonical_raw = np.concatenate(
            parts,
            axis=0,
        )

        track_directory = (
            tracks_root / track_id
        )

        track_directory.mkdir(
            parents=True,
            exist_ok=True,
        )


        canonical_cloud = (
            o3d.geometry.PointCloud()
        )
        canonical_cloud.points = (
            o3d.utility.Vector3dVector(
                canonical_raw
            )
        )

        canonical_voxel_cloud = (
            canonical_cloud.voxel_down_sample(
                args.voxel_size
            )
        )

        canonical_voxel = np.asarray(
            canonical_voxel_cloud.points,
            dtype=np.float32,
        )

        save_point_cloud(
            canonical_raw,
            track_directory
            / "canonical_raw.ply",
        )

        save_point_cloud(
            canonical_voxel,
            track_directory
            / (
                "canonical_voxel_"
                f"{args.voxel_size:.3f}.ply"
            ),
        )

        np.savez_compressed(
            track_directory
            / "canonical_geometry.npz",
            xyz_object=canonical_voxel,
        )

        track_metadata[
            track_id
        ]["raw_accumulated_points"] = int(
            len(canonical_raw)
        )

        track_metadata[
            track_id
        ]["voxelized_points"] = int(
            len(canonical_voxel)
        )

        track_metadata[
            track_id
        ]["num_observations"] = int(
            len(parts)
        )

        with (
            track_directory
            / "metadata.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                track_metadata[track_id],
                file,
                indent=2,
            )

    summary = {
        "source_tfrecord": str(
            tfrecord_path
        ),
        "start_frame": int(
            args.start_frame
        ),
        "end_frame_inclusive": int(
            end_frame_exclusive - 1
        ),
        "reference_frame": int(
            reference_index
        ),
        "lidar_names": lidar_names,
        "return_index": int(
            return_index
        ),
        "voxel_size": float(
            args.voxel_size
        ),
        "box_margin_xy": float(
            args.box_margin_xy
        ),
        "box_margin_z": float(
            args.box_margin_z
        ),
        "foreground_types": [
            "VEHICLE",
            "PEDESTRIAN",
            "CYCLIST",
        ],
        "static_raw_points": int(
            len(static_raw)
        ),
        "static_voxel_points": int(
            len(static_voxel)
        ),
        "num_tracks": int(
            len(canonical_track_parts)
        ),
        "frames": frame_summary,
    }

    with (
        output_root / "summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print("\nSaved")
    print("-----")
    print(
        f"Static raw:    "
        f"{output_root / 'static_raw.ply'}"
    )
    print(
        f"Static voxel:  "
        f"{output_root / f'static_voxel_{args.voxel_size:.3f}.ply'}"
    )
    print(
        f"Tracks:        {len(canonical_track_parts)}"
    )
    print(
        f"Summary:       "
        f"{output_root / 'summary.json'}"
    )


if __name__ == "__main__":
    main()