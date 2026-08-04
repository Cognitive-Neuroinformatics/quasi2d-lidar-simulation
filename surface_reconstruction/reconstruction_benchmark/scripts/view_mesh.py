#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import open3d as o3d


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh", type=Path)
    parser.add_argument(
        "--point-cloud",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--wireframe",
        action="store_true",
    )
    args = parser.parse_args()

    mesh = o3d.io.read_triangle_mesh(str(args.mesh))

    if mesh.is_empty():
        raise RuntimeError(f"Could not load mesh: {args.mesh}")

    mesh.compute_vertex_normals()

    print(mesh)
    print("Vertices:", len(mesh.vertices))
    print("Triangles:", len(mesh.triangles))
    print("Watertight:", mesh.is_watertight())
    print(
        "Edge manifold with boundaries:",
        mesh.is_edge_manifold(allow_boundary_edges=True),
    )
    print(
        "Edge manifold without boundaries:",
        mesh.is_edge_manifold(allow_boundary_edges=False),
    )

    geometries = [mesh]

    if args.wireframe:
        wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
        geometries.append(wireframe)

    if args.point_cloud is not None:
        cloud = o3d.io.read_point_cloud(str(args.point_cloud))
        if cloud.is_empty():
            raise RuntimeError(
                f"Could not load point cloud: {args.point_cloud}"
            )
        geometries.append(cloud)

    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=3.0,
        origin=[0.0, 0.0, 0.0],
    )
    geometries.append(coordinate_frame)

    o3d.visualization.draw_geometries(
        geometries,
        window_name=args.mesh.name,
        width=1400,
        height=900,
        mesh_show_back_face=True,
        mesh_show_wireframe=args.wireframe,
    )


if __name__ == "__main__":
    main()



# python scripts/view_mesh.py \
#   /home/cni/Documents/PhD_status/surface_reconstruction/data/outputs/meshes/frame_000_poisson_depth9.ply


# python scripts/view_mesh.py \
#   /home/cni/Documents/PhD_status/surface_reconstruction/data/outputs/meshes/frame_000_poisson_depth9.ply \
#   --point-cloud /home/cni/Documents/PhD_status/surface_reconstruction/data/extracted/frame_000_top_return_1_normals_r050.ply


# python scripts/view_mesh.py \
#   /home/cni/Documents/PhD_status/surface_reconstruction/data/outputs/meshes/frame_000_poisson_depth9.ply \
#   --wireframe