#!/usr/bin/env python3

"""
Point-Cloud Preprocessing for Surface Reconstruction
====================================================

Purpose
-------
This script prepares a Waymo point cloud for surface reconstruction using
Open3D. It removes invalid or isolated points, optionally downsamples the
cloud, estimates surface normals, orients the normals consistently, and saves
the resulting point cloud together with preprocessing metrics.

Processing Pipeline
-------------------
The script performs the following operations:

1. Load the input point cloud and associated attributes.

2. Remove points containing non-finite XYZ coordinates, intensity values,
   or sensor-origin values.

3. Optionally apply voxel-grid downsampling to reduce point density and
   produce a more uniform spatial distribution.

4. Optionally apply statistical outlier removal. A point is removed when
   the average distance to its local neighbours is substantially larger
   than the global neighbourhood-distance distribution.

5. Estimate a surface normal for every remaining point using neighbouring
   points within a configurable search radius.

6. Orient the estimated normals using one of two strategies:

   viewpoint:
       Orient every normal toward a specified LiDAR viewpoint. This method
       is suitable for a point cloud acquired from a single sensor pose.

   consistent_tangent_plane:
       Propagate consistent normal orientations through neighbouring
       points. This method is more suitable for multi-frame point clouds
       accumulated from several sensor positions.

7. Save the preprocessed point cloud as a binary PLY file.

8. Save point counts, parameter values, and runtime measurements as JSON
   metadata.

Single-Frame and Multi-Frame Usage
----------------------------------
For a single-frame Waymo cloud, normals may be oriented toward the TOP LiDAR
origin because all points share approximately the same viewpoint.

For an accumulated or composed multi-frame cloud, points originate from
several sensor poses. In that case, normals should not all be oriented toward
one viewpoint. The consistent-tangent-plane method should be used instead.

Expected Output
---------------
The resulting PLY contains:

- filtered XYZ coordinates,
- optional grayscale intensity colors,
- estimated and consistently oriented surface normals.

This preprocessed cloud can be used as input to reconstruction algorithms
such as Poisson reconstruction, Ball Pivoting, GP3, or other mesh-generation
methods.
"""


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
        input_path: Path,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray | None,
    ]:
        """
        Load either:

        1. An NPZ file containing at least:
            xyz

        Optional arrays:
            intensity
            sensor_origins

        2. A PLY point cloud containing XYZ coordinates and optionally colors.

        Returns:
            xyz:
                Shape (N, 3).

            intensity:
                Shape (N,). Zero-filled when unavailable.

            sensor_origins:
                Shape (N, 3), or None when unavailable.
        """
        if not input_path.exists():
            raise FileNotFoundError(
                f"Input file not found: {input_path}"
            )

        suffix = input_path.suffix.lower()

        if suffix == ".npz":
            data = np.load(input_path)

            if "xyz" not in data.files:
                raise KeyError(
                    "The NPZ file must contain an 'xyz' array"
                )

            xyz = np.asarray(
                data["xyz"],
                dtype=np.float64,
            )

            if "intensity" in data.files:
                intensity = np.asarray(
                    data["intensity"],
                    dtype=np.float64,
                )
            else:
                intensity = np.zeros(
                    len(xyz),
                    dtype=np.float64,
                )

            if "sensor_origins" in data.files:
                sensor_origins = np.asarray(
                    data["sensor_origins"],
                    dtype=np.float64,
                )
            else:
                sensor_origins = None

        elif suffix in {".ply", ".pcd", ".xyz", ".xyzn", ".xyzrgb"}:
            cloud = o3d.io.read_point_cloud(
                str(input_path)
            )

            if cloud.is_empty():
                raise RuntimeError(
                    f"Point cloud is empty: {input_path}"
                )

            xyz = np.asarray(
                cloud.points,
                dtype=np.float64,
            )

            if cloud.has_colors():
                colors = np.asarray(
                    cloud.colors,
                    dtype=np.float64,
                )

                # Convert RGB colors to one grayscale value.
                intensity = colors.mean(axis=1)
            else:
                intensity = np.zeros(
                    len(xyz),
                    dtype=np.float64,
                )

            sensor_origins = None

        else:
            raise ValueError(
                f"Unsupported input format: {suffix}. "
                "Supported formats include NPZ and PLY."
            )

        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(
                f"Unexpected XYZ shape: {xyz.shape}"
            )

        if intensity.ndim != 1 or len(intensity) != len(xyz):
            raise ValueError(
                "Intensity must have shape (N,), got "
                f"{intensity.shape}"
            )

        if (
            sensor_origins is not None
            and sensor_origins.shape != xyz.shape
        ):
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
    )

    if sensor_origins is not None:
        finite_mask &= np.isfinite(
            sensor_origins
        ).all(axis=1)

    xyz = xyz[finite_mask]
    intensity = intensity[finite_mask]

    if sensor_origins is not None:
        sensor_origins = sensor_origins[
            finite_mask
        ]

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


    radius_outlier_config = preprocessing.get(
        "radius_outlier_removal",
        {},
    )

    if radius_outlier_config.get(
        "enabled",
        False,
    ):
        start = time.perf_counter()

        radius_value = float(
            radius_outlier_config["radius"]
        )
        minimum_neighbors = int(
            radius_outlier_config["min_neighbors"]
        )

        cloud, retained_indices = (
            cloud.remove_radius_outlier(
                nb_points=minimum_neighbors,
                radius=radius_value,
            )
        )

        timings[
            "radius_outlier_removal_seconds"
        ] = time.perf_counter() - start

        point_counts[
            "after_radius_outlier_removal"
        ] = int(len(cloud.points))

        print(
            "After radius outlier removal: "
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
        if sensor_origins is None:
            raise ValueError(
                "Viewpoint normal orientation requires "
                "sensor_origins in the input NPZ. "
                "For multi-frame or PLY input, use "
                "'consistent_tangent_plane'."
            )

        viewpoint = np.median(
            sensor_origins,
            axis=0,
        )

        orient_normals_toward_viewpoint(
            cloud=cloud,
            viewpoint=viewpoint,
        )

        print(
            "Normals oriented toward viewpoint: "
            f"{viewpoint}"
        )

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
        "voxel_downsample_enabled": bool(
            voxel_config.get("enabled", False)
        ),
        "voxel_size": float(
            voxel_config.get("voxel_size", 0.0)
        ),
        "statistical_outlier_enabled": bool(
            outlier_config.get("enabled", False)
        ),
        "outlier_nb_neighbors": int(
            outlier_config.get(
                "nb_neighbors",
                0,
            )
        ),
        "outlier_std_ratio": float(
            outlier_config.get(
                "std_ratio",
                0.0,
            )
        ),
        "radius_outlier_enabled": bool(
            radius_outlier_config.get(
                "enabled",
                False,
            )
        ),
        "radius_outlier_radius": float(
            radius_outlier_config.get(
                "radius",
                0.0,
            )
        ),
        "radius_outlier_min_neighbors": int(
            radius_outlier_config.get(
                "min_neighbors",
                0,
            )
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