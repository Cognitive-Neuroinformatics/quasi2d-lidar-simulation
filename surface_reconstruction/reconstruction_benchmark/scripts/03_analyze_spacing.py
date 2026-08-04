#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("point_cloud", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/metrics/frame_000_spacing.json"),
    )
    args = parser.parse_args()

    cloud = o3d.io.read_point_cloud(str(args.point_cloud))

    if cloud.is_empty():
        raise RuntimeError(f"Could not read point cloud: {args.point_cloud}")

    print(f"Computing nearest-neighbour distances for {len(cloud.points):,} points")

    distances = np.asarray(
        cloud.compute_nearest_neighbor_distance(),
        dtype=np.float64,
    )

    distances = distances[np.isfinite(distances)]
    distances = distances[distances > 0.0]

    if distances.size == 0:
        raise RuntimeError("No valid nearest-neighbour distances found")

    percentiles = {
        "p01": float(np.percentile(distances, 1)),
        "p05": float(np.percentile(distances, 5)),
        "p10": float(np.percentile(distances, 10)),
        "p25": float(np.percentile(distances, 25)),
        "p50": float(np.percentile(distances, 50)),
        "p75": float(np.percentile(distances, 75)),
        "p90": float(np.percentile(distances, 90)),
        "p95": float(np.percentile(distances, 95)),
        "p99": float(np.percentile(distances, 99)),
    }

    statistics = {
        "point_cloud": str(args.point_cloud),
        "point_count": int(len(cloud.points)),
        "valid_distance_count": int(len(distances)),
        "mean": float(np.mean(distances)),
        "standard_deviation": float(np.std(distances)),
        "minimum": float(np.min(distances)),
        "maximum": float(np.max(distances)),
        "percentiles": percentiles,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as file:
        json.dump(statistics, file, indent=2)

    print("\nNearest-neighbour spacing:")
    print(f"  mean:   {statistics['mean']:.4f} m")
    print(f"  median: {percentiles['p50']:.4f} m")
    print(f"  p10:    {percentiles['p10']:.4f} m")
    print(f"  p90:    {percentiles['p90']:.4f} m")
    print(f"  p95:    {percentiles['p95']:.4f} m")
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()