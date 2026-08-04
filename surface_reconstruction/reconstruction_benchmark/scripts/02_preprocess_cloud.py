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
        raise ValueError(f"Invalid YAML file: {path}")

    return config


def load_input(
    npz_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not npz_path.exists():
        raise FileNotFoundError(f"Input file not found: {npz_path}")

    data = np.load(npz_path)

    required = {"xyz", "intensity", "sensor_origins"}
    missing = required.difference(data.files)

    if missing:
        raise KeyError(f"Missing arrays in NPZ: {sorted(missing)}")

    xyz = np.asarray(data["xyz"], dtype=np.float64)
    intensity = np.asarray(data["intensity"], dtype=np.float64)
    sensor_origins = np.asarray(
        data["sensor_origins"],
        dtype=np.float64,
    )

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"Unexpected XYZ shape: {xyz.shape}")

    if sensor_origins.shape != xyz.shape:
        raise ValueError(
            "sensor_origins must have the same shape as xyz: "
            f"{sensor_origins.shape} versus {xyz.shape}"
        )

    return xyz, intensity, sensor_origins


def make_cloud(
    xyz: np.ndarray,
    intensity: np.ndarray,
) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(xyz)

    if len(intensity) == len(xyz):
        lower = float(np.percentile(intensity, 1))
        upper = float(np.percentile(intensity, 99))

        if upper > lower:
            values = np.clip(
                (intensity - lower) / (upper - lower),
                0.0,
                1.0,
            )
        else:
            values = np.full(len(xyz), 0.5)

        colors = np.repeat(values[:, None], 3, axis=1)
        cloud.colors = o3d.utility.Vector3dVector(colors)

    return cloud


def orient_normals_toward_viewpoint(
    cloud: o3d.geometry.PointCloud,
    viewpoint: np.ndarray,
) -> None:
    """
    Orient every normal toward the LiDAR origin.

    The normal is flipped when it points away from the sensor.
    """
    points = np.asarray(cloud.points)
    normals = np.asarray(cloud.normals)

    directions_to_sensor = viewpoint[None, :] - points
    dot_products = np.sum(normals * directions_to_sensor, axis=1)

    flip_mask = dot_products < 0.0
    normals[flip_mask] *= -1.0

    cloud.normals = o3d.utility.Vector3dVector(normals)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess an extracted Waymo point cloud."
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
        help="Extracted NPZ containing XYZ and sensor origins.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/extracted/frame_000_top_return_1_preprocessed.ply"
        ),
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    preprocessing = config["preprocessing"]

    xyz, intensity, sensor_origins = load_input(args.input)

    print(f"Input points: {len(xyz):,}")

    finite_mask = (
        np.isfinite(xyz).all(axis=1)
        & np.isfinite(intensity)
        & np.isfinite(sensor_origins).all(axis=1)
    )

    xyz = xyz[finite_mask]
    intensity = intensity[finite_mask]
    sensor_origins = sensor_origins[finite_mask]

    cloud = make_cloud(xyz, intensity)

    timings: dict[str, float] = {}
    point_counts: dict[str, int] = {
        "input": int(len(cloud.points)),
    }

    voxel_config = preprocessing["voxel_downsample"]

    if voxel_config.get("enabled", False):
        start = time.perf_counter()

        voxel_size = float(voxel_config["voxel_size"])
        cloud = cloud.voxel_down_sample(voxel_size)

        timings["voxel_downsample_seconds"] = (
            time.perf_counter() - start
        )
        point_counts["after_voxel_downsample"] = int(
            len(cloud.points)
        )

        print(
            f"After {voxel_size:.3f} m voxel downsampling: "
            f"{len(cloud.points):,}"
        )

    outlier_config = preprocessing[
        "statistical_outlier_removal"
    ]

    if outlier_config.get("enabled", False):
        start = time.perf_counter()

        cloud, retained_indices = cloud.remove_statistical_outlier(
            nb_neighbors=int(outlier_config["nb_neighbors"]),
            std_ratio=float(outlier_config["std_ratio"]),
        )

        timings["outlier_removal_seconds"] = (
            time.perf_counter() - start
        )
        point_counts["after_outlier_removal"] = int(
            len(cloud.points)
        )

        print(
            "After statistical outlier removal: "
            f"{len(cloud.points):,}"
        )

    normal_config = preprocessing["normals"]

    start = time.perf_counter()

    radius = float(normal_config["radius"])
    max_nn = int(normal_config["max_nn"])

    cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius,
            max_nn=max_nn,
        ),
        fast_normal_computation=False,
    )

    cloud.normalize_normals()

    timings["normal_estimation_seconds"] = (
        time.perf_counter() - start
    )

    orientation_config = preprocessing.get(
        "orient_normals",
        {},
    )
    orientation_method = orientation_config.get(
        "method",
        "viewpoint",
    )

    if orientation_method == "viewpoint":
        # All extracted points currently come from the TOP LiDAR,
        # so their stored sensor origins are identical.
        viewpoint = np.median(sensor_origins, axis=0)

        orient_normals_toward_viewpoint(
            cloud=cloud,
            viewpoint=viewpoint,
        )

        print(f"Normals oriented toward viewpoint: {viewpoint}")

    elif orientation_method == "consistent_tangent_plane":
        k = int(orientation_config.get("k", 50))

        cloud.orient_normals_consistent_tangent_plane(k)

        print(
            "Normals oriented using consistent tangent planes "
            f"with k={k}"
        )

    else:
        raise ValueError(
            f"Unknown normal orientation method: "
            f"{orientation_method}"
        )

    if not cloud.has_normals():
        raise RuntimeError("Normal estimation failed")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    success = o3d.io.write_point_cloud(
        str(args.output),
        cloud,
        write_ascii=False,
        compressed=False,
        print_progress=True,
    )

    if not success:
        raise RuntimeError(
            f"Failed to save point cloud: {args.output}"
        )

    metrics_path = Path("outputs/metrics/preprocessing.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = {
        "input": str(args.input),
        "output": str(args.output),
        "parameters": {
            "voxel_size": float(
                voxel_config["voxel_size"]
            ),
            "outlier_nb_neighbors": int(
                outlier_config["nb_neighbors"]
            ),
            "outlier_std_ratio": float(
                outlier_config["std_ratio"]
            ),
            "normal_radius": radius,
            "normal_max_nn": max_nn,
            "normal_orientation": orientation_method,
        },
        "point_counts": point_counts,
        "timings": timings,
    }

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(f"\nSaved cloud:   {args.output}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Final points:  {len(cloud.points):,}")


if __name__ == "__main__":
    main()