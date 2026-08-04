#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("point_cloud", type=Path)
    parser.add_argument(
        "--sensor-origin",
        nargs=3,
        type=float,
        default=[1.43, 0.0, 2.184],
    )
    args = parser.parse_args()

    cloud = o3d.io.read_point_cloud(str(args.point_cloud))

    if cloud.is_empty():
        raise RuntimeError("Point cloud is empty")

    if not cloud.has_normals():
        raise RuntimeError("Point cloud has no normals")

    points = np.asarray(cloud.points)
    normals = np.asarray(cloud.normals)
    sensor = np.asarray(args.sensor_origin, dtype=np.float64)

    direction_to_sensor = sensor[None, :] - points
    orientation_dot = np.sum(normals * direction_to_sensor, axis=1)

    print("All points")
    print("----------")
    print(f"Point count:                  {len(points):,}")
    print(f"Normals toward sensor:        {(orientation_dot >= 0).sum():,}")
    print(f"Normals away from sensor:     {(orientation_dot < 0).sum():,}")
    print(
        "Percentage toward sensor:     "
        f"{100.0 * np.mean(orientation_dot >= 0):.3f}%"
    )
    print(f"Normal-z minimum:             {normals[:, 2].min():.4f}")
    print(f"Normal-z median:              {np.median(normals[:, 2]):.4f}")
    print(f"Normal-z maximum:             {normals[:, 2].max():.4f}")

    # Approximate road candidates:
    # low points, excluding the immediate ego-vehicle hole.
    radial_xy = np.linalg.norm(
    points[:, :2] - sensor[None, :2],
    axis=1,
    )

    road_mask = (
        (points[:, 2] >= -0.8)
        & (points[:, 2] <= 0.1)
        & (radial_xy > 3.0)
        & (radial_xy < 40.0)
    )

    road_points = points[road_mask]
    road_normals = normals[road_mask]
    road_dot = orientation_dot[road_mask]

    print("\nApproximate road candidates")
    print("---------------------------")
    print(f"Point count:                  {len(road_points):,}")

    if len(road_points) > 0:
        print(
            "Normals toward sensor:        "
            f"{100.0 * np.mean(road_dot >= 0):.3f}%"
        )
        print(
            "Normals with positive z:       "
            f"{100.0 * np.mean(road_normals[:, 2] > 0):.3f}%"
        )
        print(
            "Normals with negative z:       "
            f"{100.0 * np.mean(road_normals[:, 2] < 0):.3f}%"
        )
        print(
            "Median road normal:            "
            f"{np.median(road_normals, axis=0)}"
        )

        verticality = np.abs(road_normals[:, 2])

        print(
            "Strongly vertical |nz| > 0.8:  "
            f"{100.0 * np.mean(verticality > 0.8):.3f}%"
        )
        print(
            "Mostly horizontal |nz| < 0.3:  "
            f"{100.0 * np.mean(verticality < 0.3):.3f}%"
        )


if __name__ == "__main__":
    main()