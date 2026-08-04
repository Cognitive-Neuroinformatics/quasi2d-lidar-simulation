#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def load_point_cloud(path: Path) -> np.ndarray:
    """Load a PLY point cloud as an Nx3 NumPy array."""

    if not path.exists():
        raise FileNotFoundError(path)

    cloud = o3d.io.read_point_cloud(str(path))

    if cloud.is_empty():
        raise RuntimeError(f"Point cloud is empty: {path}")

    return np.asarray(
        cloud.points,
        dtype=np.float64,
    )


def save_point_cloud(
    xyz: np.ndarray,
    path: Path,
) -> None:
    """Save an Nx3 point array as a binary PLY point cloud."""

    xyz = np.asarray(xyz, dtype=np.float64)

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(
            f"Expected shape (N, 3), got {xyz.shape}"
        )

    if len(xyz) == 0:
        raise RuntimeError(
            f"Cannot save an empty cloud: {path}"
        )

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(xyz)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    success = o3d.io.write_point_cloud(
        str(path),
        cloud,
        write_ascii=False,
        compressed=False,
    )

    if not success:
        raise RuntimeError(
            f"Failed to save point cloud: {path}"
        )


def transform_points(
    xyz: np.ndarray,
    transform: np.ndarray,
) -> np.ndarray:
    """Apply a 4x4 transform to an Nx3 point array."""

    xyz = np.asarray(xyz, dtype=np.float64)
    transform = np.asarray(transform, dtype=np.float64)

    if transform.shape != (4, 4):
        raise ValueError(
            f"Expected transform shape (4, 4), got "
            f"{transform.shape}"
        )

    homogeneous = np.concatenate(
        [
            xyz,
            np.ones(
                (len(xyz), 1),
                dtype=np.float64,
            ),
        ],
        axis=1,
    )

    return (
        homogeneous @ transform.T
    )[:, :3]


def find_observation(
    metadata: dict,
    target_frame: int,
) -> dict | None:
    """Find one track observation for the requested frame."""

    for observation in metadata.get(
        "observations",
        [],
    ):
        if int(
            observation["frame_index"]
        ) == target_frame:
            return observation

    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compose one dense dynamic Waymo scene from a shared "
            "static map and canonical tracked objects."
        )
    )

    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help=(
            "Experiment root containing static_voxel_*.ply "
            "and tracks/."
        ),
    )

    parser.add_argument(
        "--target-frame",
        type=int,
        required=True,
        help="Frame whose object poses should be used.",
    )

    parser.add_argument(
        "--static-cloud",
        type=str,
        default="static_voxel_0.050.ply",
    )

    parser.add_argument(
        "--canonical-cloud",
        type=str,
        default="canonical_voxel_0.050.ply",
    )

    parser.add_argument(
        "--minimum-observations",
        type=int,
        default=2,
        help=(
            "Ignore tracks observed in fewer than this number "
            "of frames."
        ),
    )

    parser.add_argument(
        "--final-voxel-size",
        type=float,
        default=0.0,
        help=(
            "Optional final voxel downsampling in metres. "
            "Use 0 to disable."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    if args.target_frame < 0:
        raise ValueError(
            "--target-frame must be non-negative"
        )

    if args.minimum_observations <= 0:
        raise ValueError(
            "--minimum-observations must be positive"
        )

    if args.final_voxel_size < 0:
        raise ValueError(
            "--final-voxel-size must be zero or positive"
        )

    root = args.root
    tracks_root = root / "tracks"
    static_path = root / args.static_cloud

    if not tracks_root.exists():
        raise FileNotFoundError(tracks_root)

    static_xyz = load_point_cloud(
        static_path
    )

    scene_parts: list[np.ndarray] = [
        static_xyz
    ]

    placed_objects: list[dict] = []

    print(f"Target frame:  {args.target_frame}")
    print(f"Static cloud:  {static_path}")
    print(f"Static points: {len(static_xyz):,}")

    for track_directory in sorted(
        path
        for path in tracks_root.iterdir()
        if path.is_dir()
    ):
        metadata_path = (
            track_directory / "metadata.json"
        )

        canonical_path = (
            track_directory
            / args.canonical_cloud
        )

        if (
            not metadata_path.exists()
            or not canonical_path.exists()
        ):
            continue

        metadata = json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )

        num_observations = int(
            metadata.get(
                "num_observations",
                0,
            )
        )

        if (
            num_observations
            < args.minimum_observations
        ):
            continue

        observation = find_observation(
            metadata,
            args.target_frame,
        )

        # The object does not exist or is not annotated in this frame.
        if observation is None:
            continue

        if "object_to_reference" not in observation:
            raise KeyError(
                "Missing object_to_reference for track "
                f"{metadata.get('track_id')}. "
                "Rerun extract_static_and_track.py."
            )

        object_to_reference = np.asarray(
            observation["object_to_reference"],
            dtype=np.float64,
        )

        canonical_xyz = load_point_cloud(
            canonical_path
        )

        placed_xyz = transform_points(
            canonical_xyz,
            object_to_reference,
        )

        scene_parts.append(
            placed_xyz
        )

        record = {
            "track_id": metadata["track_id"],
            "type": metadata["type"],
            "num_observations": num_observations,
            "canonical_points": int(
                len(canonical_xyz)
            ),
            "placed_points": int(
                len(placed_xyz)
            ),
        }

        placed_objects.append(record)

        print(
            f"Placed {metadata['type']:10s} "
            f"{metadata['track_id']} "
            f"({len(placed_xyz):,} points)"
        )

    scene_xyz = np.concatenate(
        scene_parts,
        axis=0,
    )

    points_before_voxel = len(
        scene_xyz
    )

    if args.final_voxel_size > 0:
        scene_cloud = (
            o3d.geometry.PointCloud()
        )

        scene_cloud.points = (
            o3d.utility.Vector3dVector(
                scene_xyz
            )
        )

        scene_cloud = (
            scene_cloud.voxel_down_sample(
                args.final_voxel_size
            )
        )

        scene_xyz = np.asarray(
            scene_cloud.points,
            dtype=np.float64,
        )

    save_point_cloud(
        scene_xyz,
        args.output,
    )

    metadata_output = (
        args.output.with_suffix(".json")
    )

    output_metadata = {
        "target_frame": int(
            args.target_frame
        ),
        "static_cloud": str(
            static_path
        ),
        "canonical_cloud_name": (
            args.canonical_cloud
        ),
        "minimum_observations": int(
            args.minimum_observations
        ),
        "final_voxel_size": float(
            args.final_voxel_size
        ),
        "static_points": int(
            len(static_xyz)
        ),
        "objects_placed": int(
            len(placed_objects)
        ),
        "points_before_final_voxel": int(
            points_before_voxel
        ),
        "final_points": int(
            len(scene_xyz)
        ),
        "objects": placed_objects,
    }

    metadata_output.write_text(
        json.dumps(
            output_metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nSaved")
    print("-----")
    print(f"Scene:    {args.output}")
    print(f"Metadata: {metadata_output}")
    print(
        f"Objects:  {len(placed_objects)}"
    )
    print(
        f"Points:   {len(scene_xyz):,}"
    )


if __name__ == "__main__":
    main()