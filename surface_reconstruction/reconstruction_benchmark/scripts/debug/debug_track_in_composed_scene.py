#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def load_cloud(path: Path) -> o3d.geometry.PointCloud:
    cloud = o3d.io.read_point_cloud(str(path))

    if cloud.is_empty():
        raise RuntimeError(f"Empty point cloud: {path}")

    return cloud


def make_box_lines(
    center: np.ndarray,
    length: float,
    width: float,
    height: float,
    heading: float,
) -> o3d.geometry.LineSet:
    box = o3d.geometry.OrientedBoundingBox(
        center=center,
        R=o3d.geometry.get_rotation_matrix_from_axis_angle(
            np.array([0.0, 0.0, heading], dtype=np.float64)
        ),
        extent=np.array(
            [length, width, height],
            dtype=np.float64,
        ),
    )

    lines = o3d.geometry.LineSet.create_from_oriented_bounding_box(box)
    lines.paint_uniform_color([0.0, 0.0, 1.0])

    return lines


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--track-id",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--frame-index",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--static-cloud",
        type=str,
        default="static_voxel_0.050.ply",
    )

    parser.add_argument(
        "--canonical-cloud",
        type=str,
        default="canonical_voxel_0.050.ply",
    )

    args = parser.parse_args()

    track_dir = args.root / "tracks" / args.track_id
    metadata_path = track_dir / "metadata.json"
    canonical_path = track_dir / args.canonical_cloud
    static_path = args.root / args.static_cloud

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    observation = next(
        item
        for item in metadata["observations"]
        if int(item["frame_index"]) == args.frame_index
    )

    static_cloud = load_cloud(static_path)
    canonical_cloud = load_cloud(canonical_path)

    static_cloud.paint_uniform_color([0.65, 0.65, 0.65])
    canonical_cloud.paint_uniform_color([1.0, 0.0, 0.0])

    transform = np.asarray(
        observation["object_to_reference"],
        dtype=np.float64,
    )

    canonical_cloud.transform(transform)

    center_vehicle = np.asarray(
        observation["center_vehicle"],
        dtype=np.float64,
    )

    vehicle_to_reference = np.asarray(
        observation["vehicle_to_reference"],
        dtype=np.float64,
    )

    center_h = np.concatenate(
        [center_vehicle, [1.0]]
    )

    center_reference = (
        vehicle_to_reference @ center_h
    )[:3]

    box_lines = make_box_lines(
        center=center_reference,
        length=float(observation["length"]),
        width=float(observation["width"]),
        height=float(observation["height"]),
        heading=float(observation["heading"]),
    )

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        window_name=(
            f"Track {args.track_id} - frame {args.frame_index}"
        ),
        width=1400,
        height=900,
    )

    vis.add_geometry(static_cloud)
    vis.add_geometry(canonical_cloud)
    vis.add_geometry(box_lines)

    view = vis.get_view_control()

    eye_offset = np.array(
        [-8.0, -8.0, 5.0],
        dtype=np.float64,
    )

    eye = center_reference + eye_offset
    direction = center_reference - eye
    direction /= np.linalg.norm(direction)

    view.set_lookat(center_reference)
    view.set_front(direction)
    view.set_up([0.0, 0.0, 1.0])
    view.set_zoom(0.25)

    print("Grey: dense static map")
    print("Red: placed canonical object")
    print("Blue: current Waymo box")
    print("Mouse drag: rotate")
    print("Mouse wheel: zoom")
    print("Shift + drag: pan")

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()