#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def load_point_cloud(path: Path) -> np.ndarray:
    cloud = o3d.io.read_point_cloud(str(path))

    if cloud.is_empty():
        raise RuntimeError(f"Point cloud is empty: {path}")

    return np.asarray(cloud.points, dtype=np.float64)


def transform_points(
    xyz: np.ndarray,
    transform: np.ndarray,
) -> np.ndarray:
    xyz_h = np.concatenate(
        [
            xyz,
            np.ones((len(xyz), 1), dtype=np.float64),
        ],
        axis=1,
    )

    transformed = xyz_h @ transform.T
    return transformed[:, :3]


def save_point_cloud(
    xyz: np.ndarray,
    path: Path,
) -> None:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(xyz)

    path.parent.mkdir(parents=True, exist_ok=True)

    if not o3d.io.write_point_cloud(
        str(path),
        cloud,
        write_ascii=False,
        compressed=False,
    ):
        raise RuntimeError(f"Failed to save: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Place a canonical dynamic-object cloud back into "
            "its original per-frame vehicle poses."
        )
    )

    parser.add_argument(
        "--track-directory",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--canonical-cloud",
        type=str,
        default="canonical_voxel_0.050.ply",
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    track_directory = args.track_directory

    metadata_path = (
        track_directory / "metadata.json"
    )

    canonical_path = (
        track_directory / args.canonical_cloud
    )

    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    if not canonical_path.exists():
        raise FileNotFoundError(canonical_path)

    metadata = json.loads(
        metadata_path.read_text(
            encoding="utf-8"
        )
    )

    canonical_xyz = load_point_cloud(
        canonical_path
    )

    observations = metadata.get(
        "observations",
        [],
    )

    if not observations:
        raise RuntimeError(
            "Track metadata has no observations"
        )

    args.output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Track ID:       {metadata['track_id']}")
    print(f"Type:           {metadata['type']}")
    print(f"Canonical pts:  {len(canonical_xyz):,}")
    print(f"Observations:   {len(observations)}")

    for observation in observations:
        frame_index = int(
            observation["frame_index"]
        )

        object_to_vehicle = np.asarray(
            observation["object_to_vehicle"],
            dtype=np.float64,
        )

        if object_to_vehicle.shape != (4, 4):
            raise ValueError(
                f"Invalid transform shape for frame "
                f"{frame_index}: "
                f"{object_to_vehicle.shape}"
            )

        xyz_vehicle = transform_points(
            canonical_xyz,
            object_to_vehicle,
        )

        output_path = (
            args.output_directory
            / f"frame_{frame_index:03d}.ply"
        )

        save_point_cloud(
            xyz_vehicle,
            output_path,
        )

        center = object_to_vehicle[:3, 3]

        print(
            f"Frame {frame_index:03d}: "
            f"center=["
            f"{center[0]:.3f}, "
            f"{center[1]:.3f}, "
            f"{center[2]:.3f}] "
            f"→ {output_path}"
        )


if __name__ == "__main__":
    main()