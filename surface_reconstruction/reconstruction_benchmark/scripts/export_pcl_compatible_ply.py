#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    cloud = o3d.io.read_point_cloud(str(args.input))

    if cloud.is_empty():
        raise RuntimeError(f"Empty point cloud: {args.input}")

    if not cloud.has_normals():
        raise RuntimeError("Input point cloud has no normals")

    points = np.asarray(cloud.points, dtype=np.float32)
    normals = np.asarray(cloud.normals, dtype=np.float32)

    valid = (
        np.isfinite(points).all(axis=1)
        & np.isfinite(normals).all(axis=1)
    )

    points = points[valid]
    normals = normals[valid]

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {len(points)}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("property float normal_x\n")
        file.write("property float normal_y\n")
        file.write("property float normal_z\n")
        file.write("end_header\n")

        for point, normal in zip(points, normals):
            file.write(
                f"{point[0]:.8f} "
                f"{point[1]:.8f} "
                f"{point[2]:.8f} "
                f"{normal[0]:.8f} "
                f"{normal[1]:.8f} "
                f"{normal[2]:.8f}\n"
            )

    print(f"Saved points: {len(points):,}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
