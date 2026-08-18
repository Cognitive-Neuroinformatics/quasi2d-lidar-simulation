#!/usr/bin/env python3

import argparse
import colorsys
import os

import numpy as np
import open3d as o3d


SEMANTIC_COLORS = {
    -1: [1.00, 0.00, 1.00],  # NN-unassigned: magenta
    0:  [0.00, 1.00, 1.00],  # Waymo UNDEFINED: cyan
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
    out = np.empty(
        (len(ids), 3),
        dtype=np.float64,
    )

    for sid in np.unique(ids):
        out[
            ids == sid
        ] = SEMANTIC_COLORS.get(
            int(sid),
            [0.5, 0.5, 0.5],
        )

    return out


def instance_color(iid):
    if iid == -2:
        return np.array(
            [1.0, 0.0, 1.0],
            dtype=np.float64,
        )

    if iid < 0:
        return np.array(
            [0.72, 0.72, 0.72],
            dtype=np.float64,
        )

    hue = (
        int(iid)
        * 0.618033988749895
    ) % 1.0

    return np.asarray(
        colorsys.hsv_to_rgb(
            hue,
            0.75,
            0.95,
        ),
        dtype=np.float64,
    )


def instance_colors(ids):
    out = np.empty(
        (len(ids), 3),
        dtype=np.float64,
    )

    for iid in np.unique(ids):
        out[
            ids == iid
        ] = instance_color(
            int(iid)
        )

    return out


def distance_colors(distances):
    """
    Simple grayscale distance view:
    small distance -> dark
    large distance -> white.
    Normalized using 99th percentile to prevent a few huge outliers
    from flattening the display.
    """
    if len(distances) == 0:
        return np.empty(
            (0, 3),
            dtype=np.float64,
        )

    vmax = float(
        np.percentile(
            distances,
            99,
        )
    )

    vmax = max(
        vmax,
        1e-6,
    )

    x = np.clip(
        distances / vmax,
        0.0,
        1.0,
    )

    return np.stack(
        [x, x, x],
        axis=1,
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--npz",
        required=True,
    )

    parser.add_argument(
        "--mode",
        choices=[
            "semantic",
            "instance",
            "validity",
            "distance",
        ],
        default="semantic",
    )

    parser.add_argument(
        "--display-stride",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--point-size",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=None,
        help=(
            "Visualization-only threshold. "
            "In validity mode, points farther than this are magenta. "
            "If omitted, uses the saved nn_valid mask."
        ),
    )

    args = parser.parse_args()

    data = np.load(
        args.npz
    )

    xyz = data[
        "xyz"
    ]

    semantic_id = data[
        "semantic_id"
    ]

    instance_id = data[
        "instance_id"
    ]

    nn_distance = data[
        "nn_distance"
    ]

    nn_valid = data[
        "nn_valid"
    ]

    if (
        args.distance_threshold
        is not None
    ):
        valid = (
            nn_distance
            <= args.distance_threshold
        )
    else:
        valid = nn_valid

    stride = max(
        1,
        int(
            args.display_stride
        ),
    )

    xyz = xyz[
        ::stride
    ]

    semantic_id = semantic_id[
        ::stride
    ]

    instance_id = instance_id[
        ::stride
    ]

    nn_distance = nn_distance[
        ::stride
    ]

    valid = valid[
        ::stride
    ]

    if args.mode == "semantic":

        colors = semantic_colors(
            semantic_id
        )

    elif args.mode == "instance":

        colors = instance_colors(
            instance_id
        )

    elif args.mode == "validity":

        # Valid = green
        # Invalid = bright magenta
        colors = np.tile(
            np.asarray(
                [[0.2, 0.8, 0.2]],
                dtype=np.float64,
            ),
            (len(xyz), 1),
        )

        colors[
            ~valid
        ] = [
            1.0,
            0.0,
            1.0,
        ]

    else:

        colors = distance_colors(
            nn_distance
        )

    print(
        "Displayed points:",
        f"{len(xyz):,}",
    )

    print(
        "Mode:",
        args.mode,
    )

    print(
        "Valid:",
        f"{np.count_nonzero(valid):,}",
    )

    print(
        "Invalid:",
        f"{np.count_nonzero(~valid):,}",
    )

    print(
        "NN distance median:",
        float(
            np.median(
                nn_distance
            )
        ),
        "m",
    )

    print(
        "NN distance 95%:",
        float(
            np.percentile(
                nn_distance,
                95,
            )
        ),
        "m",
    )

    pcd = o3d.geometry.PointCloud()

    pcd.points = (
        o3d.utility.Vector3dVector(
            xyz.astype(
                np.float64
            )
        )
    )

    pcd.colors = (
        o3d.utility.Vector3dVector(
            colors.astype(
                np.float64
            )
        )
    )

    vis = o3d.visualization.Visualizer()

    vis.create_window(
        window_name=(
            "Dense static NN association | "
            + args.mode
        ),
        width=1600,
        height=900,
    )

    vis.add_geometry(
        pcd
    )

    render = vis.get_render_option()
    render.point_size = args.point_size

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()