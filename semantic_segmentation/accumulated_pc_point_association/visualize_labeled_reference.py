#!/usr/bin/env python3

import argparse
import colorsys
import os

import numpy as np
import open3d as o3d


SEMANTIC_COLORS = {
    0:  [0.00, 1.00, 1.00],  # UNDEFINED - bright cyan
    1:  [0.90, 0.10, 0.10],
    2:  [0.75, 0.20, 0.10],
    3:  [0.65, 0.10, 0.20],
    4:  [0.55, 0.20, 0.20],
    5:  [1.00, 0.40, 0.00],
    6:  [1.00, 0.65, 0.00],
    7:  [0.95, 0.20, 0.70],
    8:  [0.80, 0.80, 0.10],
    9:  [1.00, 0.85, 0.00],
    10: [0.45, 0.45, 0.45],
    11: [1.00, 0.45, 0.00],
    12: [0.20, 0.70, 0.90],
    13: [0.15, 0.45, 0.90],
    14: [0.55, 0.55, 0.75],
    15: [0.20, 0.65, 0.20],
    16: [0.35, 0.25, 0.10],
    17: [0.70, 0.70, 0.70],
    18: [0.35, 0.35, 0.35],
    19: [0.95, 0.95, 0.20],
    20: [0.60, 0.50, 0.40],
    21: [0.55, 0.75, 0.55],
    22: [0.65, 0.65, 0.50],
}


def semantic_colors(ids):
    out = np.empty((len(ids), 3), dtype=np.float64)

    for sid in np.unique(ids):
        out[ids == sid] = SEMANTIC_COLORS.get(
            int(sid),
            [0.5, 0.5, 0.5],
        )

    return out


def instance_color(instance_id):
    if instance_id < 0:
        return np.array(
            [0.72, 0.72, 0.72],
            dtype=np.float64,
        )

    hue = ((int(instance_id) * 0.618033988749895) % 1.0)
    return np.asarray(
        colorsys.hsv_to_rgb(hue, 0.75, 0.95),
        dtype=np.float64,
    )


def instance_colors(ids):
    out = np.empty((len(ids), 3), dtype=np.float64)

    for iid in np.unique(ids):
        out[ids == iid] = instance_color(int(iid))

    return out


def make_cloud(xyz, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        xyz.astype(np.float64)
    )
    pcd.colors = o3d.utility.Vector3dVector(
        colors.astype(np.float64)
    )
    return pcd


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--association-root", required=True)

    parser.add_argument(
        "--frame",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--accumulated",
        action="store_true",
    )

    parser.add_argument(
        "--mode",
        choices=["semantic", "instance", "dynamic"],
        default="semantic",
    )

    parser.add_argument(
        "--world",
        action="store_true",
    )

    parser.add_argument(
        "--display-stride",
        type=int,
        default=1,
    )

    args = parser.parse_args()

    if args.accumulated:
        path = os.path.join(
            args.association_root,
            "02_accumulated_labeled",
            "labeled_static_accumulated.npz",
        )

        data = np.load(path)

        xyz = data["xyz_world"]
        semantic_id = data["semantic_id"]
        instance_id = data["instance_id"]
        dynamic_mask = np.zeros(len(xyz), dtype=bool)

        title = "Sparse accumulated labeled STATIC world"

    else:
        if args.frame is None:
            raise ValueError(
                "Use either --frame <index> or --accumulated."
            )

        path = os.path.join(
            args.association_root,
            "01_labeled_frames",
            f"frame_{args.frame:03d}.npz",
        )

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No extracted labeled frame found:\n{path}"
            )

        data = np.load(path)

        xyz = (
            data["xyz_world"]
            if args.world
            else data["xyz_vehicle"]
        )

        semantic_id = data["semantic_id"]
        instance_id = data["instance_id"]
        dynamic_mask = data["is_dynamic_track_point"]

        title = (
            f"Labeled frame {args.frame:03d} | "
            + ("world" if args.world else "vehicle")
            + " coordinates"
        )

    stride = max(1, int(args.display_stride))

    xyz = xyz[::stride]
    semantic_id = semantic_id[::stride]
    instance_id = instance_id[::stride]
    dynamic_mask = dynamic_mask[::stride]

    if args.mode == "semantic":
        colors = semantic_colors(semantic_id)

    elif args.mode == "instance":
        colors = instance_colors(instance_id)

    else:
        colors = np.tile(
            np.array([[0.65, 0.65, 0.65]], dtype=np.float64),
            (len(xyz), 1),
        )
        colors[dynamic_mask] = [1.0, 0.0, 0.0]

    print("File:")
    print(path)
    print("Displayed points:", len(xyz))
    print("Mode:", args.mode)

    print()
    print("Semantic IDs present:")
    ids, counts = np.unique(
        semantic_id,
        return_counts=True,
    )

    for sid, count in zip(ids, counts):
        print(f"  {int(sid):>3}: {int(count):,}")

    print()
    print("Unique instance IDs:", len(np.unique(instance_id)))
    print(
        "Dynamic-track points:",
        int(np.count_nonzero(dynamic_mask)),
    )

    pcd = make_cloud(xyz, colors)

    vis = o3d.visualization.Visualizer()

    vis.create_window(
        window_name=title,
        width=1600,
        height=900,
    )

    vis.add_geometry(pcd)

    render = vis.get_render_option()
    render.point_size = 1.0

    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()