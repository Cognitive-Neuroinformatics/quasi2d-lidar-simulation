#!/usr/bin/env python3
"""Compare an original static/object point cloud with its MLS reconstruction."""

from __future__ import annotations

import argparse
import colorsys
import json
from pathlib import Path

import numpy as np


SEMANTIC_COLORS = {
    1: [1.0, 0.0, 0.0], 2: [0.8, 0.2, 0.0], 3: [0.7, 0.0, 0.2],
    4: [0.6, 0.2, 0.2], 5: [1.0, 0.4, 0.0], 6: [1.0, 0.7, 0.0],
    7: [1.0, 0.0, 0.8], 8: [0.9, 0.9, 0.1], 9: [1.0, 0.5, 0.5],
    10: [0.5, 0.5, 0.5], 11: [1.0, 0.3, 0.0], 12: [0.0, 0.7, 0.9],
    13: [0.2, 0.4, 1.0], 14: [0.55, 0.35, 0.2], 15: [0.0, 0.65, 0.0],
    16: [0.35, 0.2, 0.1], 17: [0.4, 0.4, 0.8], 18: [0.25, 0.25, 0.25],
    19: [1.0, 1.0, 1.0], 20: [0.55, 0.5, 0.4], 21: [0.7, 0.6, 0.5],
    22: [0.55, 0.55, 0.55],
}


def require_open3d():
    try:
        import open3d as o3d
    except ImportError as error:
        raise RuntimeError("Install open3d to use this viewer") from error
    return o3d


def sample_indices(count: int, maximum: int, rng) -> np.ndarray:
    if maximum <= 0 or count <= maximum:
        return np.arange(count, dtype=np.int64)
    return np.sort(rng.choice(count, size=maximum, replace=False))


def load_arrays(path: Path, maximum: int, rng) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        indices = sample_indices(len(data["xyz"]), maximum, rng)
        return {
            "xyz": np.asarray(data["xyz"])[indices],
            "semantic_id": np.asarray(data["semantic_id"])[indices],
            "ground_id": np.asarray(data["ground_id"])[indices],
            "instance_id": np.asarray(data["instance_id"])[indices],
        }


def load_static_mls(manifest_path: Path, maximum: int, rng):
    with manifest_path.open() as stream:
        manifest = json.load(stream)
    total = sum(int(tile["output_points"]) for tile in manifest["tiles"])
    parts = []
    for tile in manifest["tiles"]:
        count = int(tile["output_points"])
        allocation = count if maximum <= 0 else max(1, int(round(maximum * count / total)))
        parts.append(load_arrays(manifest_path.parent / tile["file"], allocation, rng))
    return {
        name: np.concatenate([part[name] for part in parts])
        for name in parts[0]
    }


def instance_color(instance_id: int):
    if instance_id <= 0:
        return np.array([0.35, 0.35, 0.35])
    hue = (instance_id * 0.618033988749895) % 1.0
    return np.asarray(colorsys.hsv_to_rgb(hue, 0.75, 0.95))


def attribute_colors(arrays: dict[str, np.ndarray], mode: str):
    count = len(arrays["xyz"])
    if mode == "uniform":
        return np.tile([0.05, 0.35, 0.95], (count, 1))
    if mode == "semantic":
        return np.asarray(
            [SEMANTIC_COLORS.get(int(value), [0.15, 0.15, 0.15]) for value in arrays["semantic_id"]]
        )
    if mode == "ground":
        palette = {-1: [0.2, 0.2, 0.2], 0: [0.95, 0.25, 0.15], 1: [0.1, 0.75, 0.2]}
        return np.asarray([palette[int(value)] for value in arrays["ground_id"]])
    return np.asarray([instance_color(int(value)) for value in arrays["instance_id"]])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--caseid", required=True)
    parser.add_argument("--reconstruction-root", type=Path, default=None)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--static", action="store_true")
    target.add_argument("--object-id", type=int)
    parser.add_argument("--max-points", type=int, default=1000000)
    parser.add_argument("--point-size", type=float, default=1.5)
    return parser.parse_args()


def main():
    args = parse_args()
    o3d = require_open3d()
    dataset_root = args.dataset_root.resolve()
    reconstruction_root = (
        args.reconstruction_root.resolve()
        if args.reconstruction_root is not None
        else dataset_root / "mls_baseline" / args.caseid
    )
    rng = np.random.default_rng(13)

    if args.static:
        original_path = dataset_root / "recon_related" / args.caseid / "static_recon_labels.npz"
        original = load_arrays(original_path, args.max_points, rng)
        mls = load_static_mls(
            reconstruction_root / "static_manifest.json", args.max_points, rng
        )
        title = f"Static MLS comparison: {args.caseid}"
    else:
        original_path = (
            dataset_root / "temp" / args.caseid / "occ" / "preproc" / "dynamic"
            / "objects" / str(args.object_id) / "stitch_labeled.npz"
        )
        mls_path = reconstruction_root / "dynamic_objects" / str(args.object_id) / "mls_surface.npz"
        original = load_arrays(original_path, args.max_points, rng)
        mls = load_arrays(mls_path, args.max_points, rng)
        title = f"Dynamic object {args.object_id} MLS comparison"

    state = {"dataset": "mls", "mode": "semantic"}
    cloud = o3d.geometry.PointCloud()

    def update(vis):
        if state["dataset"] == "original":
            xyz = original["xyz"]
            colors = attribute_colors(original, state["mode"])
        elif state["dataset"] == "mls":
            xyz = mls["xyz"]
            colors = attribute_colors(mls, state["mode"])
        else:
            xyz = np.concatenate([original["xyz"], mls["xyz"]])
            colors = np.concatenate([
                np.tile([1.0, 0.35, 0.05], (len(original["xyz"]), 1)),
                np.tile([0.0, 0.65, 1.0], (len(mls["xyz"]), 1)),
            ])
        cloud.points = o3d.utility.Vector3dVector(np.asarray(xyz, dtype=np.float64))
        cloud.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
        vis.update_geometry(cloud)
        print(
            f"view={state['dataset']}, color={state['mode']}, points={len(xyz):,} | "
            "1 original, 2 MLS, 3 both, S semantic, G ground, I instance, U uniform"
        )
        return False

    def select_dataset(name):
        return lambda vis: (state.update(dataset=name), update(vis))[1]

    def select_mode(name):
        return lambda vis: (state.update(mode=name), update(vis))[1]

    visualizer = o3d.visualization.VisualizerWithKeyCallback()
    visualizer.create_window(window_name=title, width=1400, height=900)
    visualizer.add_geometry(cloud)
    visualizer.register_key_callback(ord("1"), select_dataset("original"))
    visualizer.register_key_callback(ord("2"), select_dataset("mls"))
    visualizer.register_key_callback(ord("3"), select_dataset("both"))
    visualizer.register_key_callback(ord("S"), select_mode("semantic"))
    visualizer.register_key_callback(ord("G"), select_mode("ground"))
    visualizer.register_key_callback(ord("I"), select_mode("instance"))
    visualizer.register_key_callback(ord("U"), select_mode("uniform"))
    options = visualizer.get_render_option()
    options.background_color = np.asarray([1.0, 1.0, 1.0])
    options.point_size = args.point_size
    update(visualizer)
    visualizer.reset_view_point(True)
    visualizer.run()
    visualizer.destroy_window()


if __name__ == "__main__":
    main()
