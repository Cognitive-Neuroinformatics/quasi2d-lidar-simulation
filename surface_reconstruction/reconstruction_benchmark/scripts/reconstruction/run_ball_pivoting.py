#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML configuration: {path}")

    return config


def cleanup_mesh(
    mesh: o3d.geometry.TriangleMesh,
) -> o3d.geometry.TriangleMesh:
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


def mesh_statistics(
    mesh: o3d.geometry.TriangleMesh,
) -> dict[str, Any]:
    return {
        "vertices": int(len(mesh.vertices)),
        "triangles": int(len(mesh.triangles)),
        "is_edge_manifold_allow_boundary": bool(
            mesh.is_edge_manifold(allow_boundary_edges=True)
        ),
        "is_edge_manifold_no_boundary": bool(
            mesh.is_edge_manifold(allow_boundary_edges=False)
        ),
        "is_vertex_manifold": bool(mesh.is_vertex_manifold()),
        "is_self_intersecting": bool(mesh.is_self_intersecting()),
        "is_watertight": bool(mesh.is_watertight()),
        "is_orientable": bool(mesh.is_orientable()),
        "min_bound": mesh.get_min_bound().tolist(),
        "max_bound": mesh.get_max_bound().tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Open3D Ball Pivoting reconstruction."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/meshes/frame_000_bpa.ply"),
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    bpa_config = config["reconstruction"]["ball_pivoting"]

    if not bpa_config.get("enabled", True):
        raise RuntimeError("Ball Pivoting is disabled in the YAML")

    cloud = o3d.io.read_point_cloud(str(args.input))

    if cloud.is_empty():
        raise RuntimeError(f"Input cloud is empty: {args.input}")

    if not cloud.has_normals():
        raise RuntimeError(
            "Ball Pivoting requires a point cloud with normals"
        )

    radii = [float(value) for value in bpa_config["radii"]]

    if not radii:
        raise ValueError("At least one BPA radius is required")

    if any(radius <= 0 for radius in radii):
        raise ValueError("All BPA radii must be positive")

    radii = sorted(radii)

    print(f"Input cloud:  {args.input}")
    print(f"Input points: {len(cloud.points):,}")
    print(f"Has normals:  {cloud.has_normals()}")
    print(f"BPA radii:    {radii}")

    start = time.perf_counter()

    mesh = (
        o3d.geometry.TriangleMesh
        .create_from_point_cloud_ball_pivoting(
            cloud,
            o3d.utility.DoubleVector(radii),
        )
    )

    runtime = time.perf_counter() - start

    raw_vertices = len(mesh.vertices)
    raw_triangles = len(mesh.triangles)

    print("\nRaw BPA result")
    print("--------------")
    print(f"Vertices:  {raw_vertices:,}")
    print(f"Triangles: {raw_triangles:,}")
    print(f"Runtime:   {runtime:.3f} seconds")

    mesh = cleanup_mesh(mesh)

    print("\nCleaned BPA result")
    print("------------------")
    print(f"Vertices:  {len(mesh.vertices):,}")
    print(f"Triangles: {len(mesh.triangles):,}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    success = o3d.io.write_triangle_mesh(
        str(args.output),
        mesh,
        write_ascii=False,
        compressed=False,
        write_vertex_normals=True,
    )

    if not success:
        raise RuntimeError(f"Failed to save mesh: {args.output}")

    metrics = {
        "method": "open3d_ball_pivoting",
        "input": str(args.input),
        "output": str(args.output),
        "parameters": {
            "radii": radii,
        },
        "runtime_seconds": runtime,
        "raw_vertices": int(raw_vertices),
        "raw_triangles": int(raw_triangles),
        "mesh": mesh_statistics(mesh),
    }

    metrics_path = Path("outputs/metrics/frame_000_bpa.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print("\nSaved")
    print("-----")
    print(f"Mesh:    {args.output}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()