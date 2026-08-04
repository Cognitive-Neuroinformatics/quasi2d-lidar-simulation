#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import open3d as o3d


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("point_cloud", type=Path)
    args = parser.parse_args()

    cloud = o3d.io.read_point_cloud(str(args.point_cloud))

    if cloud.is_empty():
        raise RuntimeError(f"Could not read point cloud: {args.point_cloud}")

    print(cloud)
    print("Points:", len(cloud.points))
    print("Has colors:", cloud.has_colors())
    print("Has normals:", cloud.has_normals())
    print("Bounds min:", cloud.get_min_bound())
    print("Bounds max:", cloud.get_max_bound())

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=3.0,
        origin=[0.0, 0.0, 0.0],
    )

    o3d.visualization.draw_geometries(
        [cloud, frame],
        window_name="Waymo TOP LiDAR — Frame 0",
        width=1400,
        height=900,
        point_show_normal=False,
    )


if __name__ == "__main__":
    main()