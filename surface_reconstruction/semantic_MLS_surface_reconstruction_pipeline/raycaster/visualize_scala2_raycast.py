#!/usr/bin/env python3
"""Interactive viewer for the MLS SCALA2 raycast sequence."""

from __future__ import annotations

import argparse
import colorsys
from pathlib import Path

import numpy as np

from visualize_mls_surface import SEMANTIC_COLORS


def require_open3d():
    try:
        import open3d as o3d
    except ImportError as error:
        raise RuntimeError("Install open3d to use this viewer") from error
    return o3d


def instance_color(value: int):
    if value <= 0:
        return [0.35, 0.35, 0.35]
    return colorsys.hsv_to_rgb((value * 0.618033988749895) % 1.0, 0.8, 0.95)


def colors(data, mode):
    count = len(data["xyz"])
    if mode == "uniform":
        return np.tile([0.05, 0.35, 0.95], (count, 1))
    if mode == "semantic":
        return np.asarray([
            SEMANTIC_COLORS.get(int(value), [0.15, 0.15, 0.15])
            for value in data["semantic_id"]
        ])
    if mode == "ground":
        palette = {-1: [0.2, 0.2, 0.2], 0: [0.95, 0.25, 0.15], 1: [0.1, 0.75, 0.2]}
        return np.asarray([palette[int(value)] for value in data["ground_id"]])
    if mode == "instance":
        return np.asarray([instance_color(int(value)) for value in data["instance_id"]])
    palette = {0: [0.1, 0.35, 0.95], 1: [1.0, 0.2, 0.05]}
    return np.asarray([palette.get(int(value), [0.2, 0.2, 0.2]) for value in data["source_type"]])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--point-size", type=float, default=3.0)
    return parser.parse_args()


def main():
    args = parse_args()
    o3d = require_open3d()
    files = sorted(args.input_dir.glob("*.npz"), key=lambda path: int(path.stem))
    if not files:
        raise FileNotFoundError(f"No NPZ frames in {args.input_dir}")
    state = {"index": 0, "mode": "semantic"}
    cloud = o3d.geometry.PointCloud()

    def update(vis, reset=False):
        path = files[state["index"]]
        with np.load(path, allow_pickle=False) as loaded:
            data = {name: np.asarray(loaded[name]) for name in (
                "xyz", "semantic_id", "ground_id", "instance_id", "source_type",
                "mirror_side", "range",
            )}
        cloud.points = o3d.utility.Vector3dVector(data["xyz"].astype(np.float64))
        cloud.colors = o3d.utility.Vector3dVector(colors(data, state["mode"]).astype(np.float64))
        vis.update_geometry(cloud)
        if reset:
            vis.reset_view_point(True)
        mirror = int(data["mirror_side"][0]) if len(data["mirror_side"]) else -1
        print(
            f"Frame {path.stem} ({state['index'] + 1}/{len(files)}), MS{mirror}, "
            f"view={state['mode']}, hits={len(data['xyz']):,}, "
            f"range={np.nanmin(data['range']):.2f}..{np.nanmax(data['range']):.2f} m | "
            "N/Right next, P/Left previous, S semantic, G ground, I instance, "
            "O source, U uniform"
        )
        return False

    def move(delta):
        def callback(vis):
            state["index"] = (state["index"] + delta) % len(files)
            return update(vis)
        return callback

    def mode(name):
        def callback(vis):
            state["mode"] = name
            return update(vis)
        return callback

    visualizer = o3d.visualization.VisualizerWithKeyCallback()
    visualizer.create_window(window_name="SCALA2 MLS raycast", width=1400, height=900)
    visualizer.add_geometry(cloud)
    visualizer.register_key_callback(ord("N"), move(1))
    visualizer.register_key_callback(262, move(1))
    visualizer.register_key_callback(ord("P"), move(-1))
    visualizer.register_key_callback(263, move(-1))
    visualizer.register_key_callback(ord("S"), mode("semantic"))
    visualizer.register_key_callback(ord("G"), mode("ground"))
    visualizer.register_key_callback(ord("I"), mode("instance"))
    visualizer.register_key_callback(ord("O"), mode("source"))
    visualizer.register_key_callback(ord("U"), mode("uniform"))
    options = visualizer.get_render_option()
    options.background_color = np.asarray([1.0, 1.0, 1.0])
    options.point_size = args.point_size
    update(visualizer, reset=True)
    visualizer.run()
    visualizer.destroy_window()


if __name__ == "__main__":
    main()
