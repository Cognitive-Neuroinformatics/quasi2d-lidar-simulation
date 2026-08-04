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


def mesh_statistics(
    mesh: o3d.geometry.TriangleMesh,
) -> dict[str, Any]:
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)

    return {
        "vertices": int(len(vertices)),
        "triangles": int(len(triangles)),
        "has_vertex_normals": bool(mesh.has_vertex_normals()),
        "has_vertex_colors": bool(mesh.has_vertex_colors()),
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


def remove_low_density_vertices(
    mesh: o3d.geometry.TriangleMesh,
    densities: np.ndarray,
    quantile: float,
) -> tuple[o3d.geometry.TriangleMesh, float, int]:
    if not 0.0 <= quantile < 1.0:
        raise ValueError(
            f"density_quantile must be in [0, 1), got {quantile}"
        )

    threshold = float(np.quantile(densities, quantile))
    removal_mask = densities < threshold
    removed_count = int(np.count_nonzero(removal_mask))

    filtered_mesh = o3d.geometry.TriangleMesh(mesh)
    filtered_mesh.remove_vertices_by_mask(removal_mask)

    return filtered_mesh, threshold, removed_count


def cleanup_mesh(
    mesh: o3d.geometry.TriangleMesh,
) -> o3d.geometry.TriangleMesh:
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()

    return mesh


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Open3D Poisson surface reconstruction."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Preprocessed PLY with oriented normals.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/meshes/frame_000_poisson_depth9.ply"
        ),
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    poisson_config = config["reconstruction"]["poisson"]

    if not poisson_config.get("enabled", True):
        raise RuntimeError("Poisson reconstruction is disabled in YAML")

    cloud = o3d.io.read_point_cloud(str(args.input))

    if cloud.is_empty():
        raise RuntimeError(f"Input point cloud is empty: {args.input}")

    if not cloud.has_normals():
        raise RuntimeError(
            "Input cloud has no normals. Use the preprocessed PLY."
        )

    print(f"Input cloud: {args.input}")
    print(f"Input points: {len(cloud.points):,}")
    print(f"Has normals: {cloud.has_normals()}")

    depth = int(poisson_config.get("depth", 9))
    width = float(poisson_config.get("width", 0.0))
    scale = float(poisson_config.get("scale", 1.1))
    linear_fit = bool(poisson_config.get("linear_fit", False))
    density_quantile = float(
        poisson_config.get("density_quantile", 0.02)
    )

    print("\nPoisson parameters")
    print("------------------")
    print(f"depth:             {depth}")
    print(f"width:             {width}")
    print(f"scale:             {scale}")
    print(f"linear_fit:        {linear_fit}")
    print(f"density_quantile:  {density_quantile}")

    start = time.perf_counter()

    mesh, densities_vector = (
        o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            cloud,
            depth=depth,
            width=width,
            scale=scale,
            linear_fit=linear_fit,
        )
    )

    reconstruction_seconds = time.perf_counter() - start

    densities = np.asarray(densities_vector, dtype=np.float64)

    print("\nRaw Poisson result")
    print("------------------")
    print(f"Vertices: {len(mesh.vertices):,}")
    print(f"Triangles: {len(mesh.triangles):,}")
    print(
        "Density range: "
        f"{densities.min():.6f} to {densities.max():.6f}"
    )

    raw_output = args.output.with_name(
        args.output.stem + "_raw" + args.output.suffix
    )
    raw_output.parent.mkdir(parents=True, exist_ok=True)

    raw_mesh = o3d.geometry.TriangleMesh(mesh)
    raw_mesh.compute_vertex_normals()

    if not o3d.io.write_triangle_mesh(
        str(raw_output),
        raw_mesh,
        write_ascii=False,
        compressed=False,
        write_vertex_normals=True,
    ):
        raise RuntimeError(f"Failed to save raw mesh: {raw_output}")

    mesh, density_threshold, removed_vertices = (
        remove_low_density_vertices(
            mesh=mesh,
            densities=densities,
            quantile=density_quantile,
        )
    )

    mesh = cleanup_mesh(mesh)

    print("\nFiltered Poisson result")
    print("-----------------------")
    print(f"Density threshold: {density_threshold:.6f}")
    print(f"Removed vertices:  {removed_vertices:,}")
    print(f"Vertices:          {len(mesh.vertices):,}")
    print(f"Triangles:         {len(mesh.triangles):,}")
    print(
        f"Reconstruction time: {reconstruction_seconds:.3f} seconds"
    )

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
        "method": "open3d_poisson",
        "input": str(args.input),
        "raw_output": str(raw_output),
        "filtered_output": str(args.output),
        "parameters": {
            "depth": depth,
            "width": width,
            "scale": scale,
            "linear_fit": linear_fit,
            "density_quantile": density_quantile,
        },
        "runtime_seconds": reconstruction_seconds,
        "density": {
            "minimum": float(densities.min()),
            "maximum": float(densities.max()),
            "mean": float(densities.mean()),
            "median": float(np.median(densities)),
            "threshold": density_threshold,
            "removed_vertices": removed_vertices,
        },
        "mesh": mesh_statistics(mesh),
    }

    metrics_path = Path(
        "outputs/metrics/frame_000_poisson_depth9.json"
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print("\nSaved")
    print("-----")
    print(f"Raw mesh:      {raw_output}")
    print(f"Filtered mesh: {args.output}")
    print(f"Metrics:       {metrics_path}")


if __name__ == "__main__":
    main()