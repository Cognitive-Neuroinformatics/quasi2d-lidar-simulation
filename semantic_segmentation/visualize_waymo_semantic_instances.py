#!/usr/bin/env python3
"""Interactive frame-by-frame viewer for propagated Waymo TOP-LiDAR labels.

Keys
----
Right arrow / N : next frame
Left arrow  / P : previous frame
S               : semantic colours
I               : instance colours
D               : static/dynamic colours
G               : ground/non-ground colours
U               : toggle unassigned/undefined (semantic IDs -1 and 0)
R               : reset camera to the current frame
K               : save a screenshot of the current view
Q / Escape      : close the viewer
"""

from __future__ import annotations

import argparse
import colorsys
import glob
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import open3d as o3d


# Established Waymo LiDAR segmentation palette. Values are RGB in [0, 1].
SEMANTIC_COLORS = {
    -1: [1.00, 0.00, 1.00],  # NN_UNASSIGNED
    0:  [0.00, 1.00, 1.00],  # UNDEFINED
    1:  [0.90, 0.10, 0.10],  # CAR
    2:  [0.75, 0.20, 0.10],  # TRUCK
    3:  [0.65, 0.10, 0.20],  # BUS
    4:  [0.55, 0.20, 0.20],  # OTHER_VEHICLE
    5:  [1.00, 0.40, 0.00],  # MOTORCYCLIST
    6:  [1.00, 0.65, 0.00],  # BICYCLIST
    7:  [0.95, 0.20, 0.70],  # PEDESTRIAN
    8:  [0.80, 0.80, 0.10],  # SIGN
    9:  [1.00, 0.85, 0.00],  # TRAFFIC_LIGHT
    10: [0.45, 0.45, 0.45],  # POLE
    11: [1.00, 0.45, 0.00],  # CONSTRUCTION_CONE
    12: [0.20, 0.70, 0.90],  # BICYCLE
    13: [0.15, 0.45, 0.90],  # MOTORCYCLE
    14: [0.55, 0.55, 0.75],  # BUILDING
    15: [0.20, 0.65, 0.20],  # VEGETATION
    16: [0.35, 0.25, 0.10],  # TREE_TRUNK
    17: [0.70, 0.70, 0.70],  # CURB
    18: [0.35, 0.35, 0.35],  # ROAD
    19: [0.95, 0.95, 0.20],  # LANE_MARKER
    20: [0.60, 0.50, 0.40],  # OTHER_GROUND
    21: [0.55, 0.75, 0.55],  # WALKABLE
    22: [0.65, 0.65, 0.50],  # SIDEWALK
}

SEMANTIC_NAMES = {
    -1: "NN_UNASSIGNED",
    0: "UNDEFINED",
    1: "CAR",
    2: "TRUCK",
    3: "BUS",
    4: "OTHER_VEHICLE",
    5: "MOTORCYCLIST",
    6: "BICYCLIST",
    7: "PEDESTRIAN",
    8: "SIGN",
    9: "TRAFFIC_LIGHT",
    10: "POLE",
    11: "CONSTRUCTION_CONE",
    12: "BICYCLE",
    13: "MOTORCYCLE",
    14: "BUILDING",
    15: "VEGETATION",
    16: "TREE_TRUNK",
    17: "CURB",
    18: "ROAD",
    19: "LANE_MARKER",
    20: "OTHER_GROUND",
    21: "WALKABLE",
    22: "SIDEWALK",
}

# Ground definition requested for binary inspection. UNDEFINED (0) and
# NN_UNASSIGNED (-1) belong to neither group.
GROUND_CLASSES = {17, 18, 19, 20, 21, 22}

# Binary ground-inspection palette. The two unknown states deliberately keep
# the same colours used by the full semantic palette.
GROUND_VIEW_COLORS = {
    "nn_unassigned": [1.00, 0.00, 1.00],  # semantic_id == -1: magenta
    "undefined":     [0.00, 1.00, 1.00],  # semantic_id == 0: cyan
    "non_ground":    [0.15, 0.45, 0.95],  # semantic_id 1..16: blue
    "ground":        [0.15, 0.80, 0.20],  # semantic_id 17..22: green
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing frame_000.npz, frame_001.npz, ...",
    )
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument(
        "--coordinate-frame",
        choices=("vehicle", "world"),
        default="vehicle",
        help="Vehicle coordinates keep the ego vehicle stationary. World "
             "coordinates show the accumulated-map alignment.",
    )
    parser.add_argument(
        "--initial-color-mode",
        choices=("semantic", "instance", "dynamic", "ground"),
        default="semantic",
    )
    parser.add_argument(
        "--hide-undefined",
        action="store_true",
        help="Initially hide NN-unassigned (-1) and Waymo UNDEFINED (0).",
    )
    parser.add_argument(
        "--screenshot-dir",
        default="viewer_screenshots",
    )
    return parser.parse_args()


def frame_number(path: str) -> int:
    stem = Path(path).stem
    try:
        return int(stem.rsplit("_", 1)[1])
    except (IndexError, ValueError) as error:
        raise ValueError(
            f"Expected a filename such as frame_025.npz, got {path}"
        ) from error


def discover_frames(input_dir: str) -> List[str]:
    paths = glob.glob(str(Path(input_dir) / "frame_*.npz"))
    paths.sort(key=frame_number)
    if not paths:
        raise FileNotFoundError(
            f"No frame_*.npz files found in {input_dir}"
        )
    return paths


def instance_color(instance_id: int) -> np.ndarray:
    if instance_id == 0:
        return np.asarray([0.55, 0.55, 0.55])
    # Golden-ratio hue spacing gives deterministic, well-separated colours.
    hue = (instance_id * 0.618033988749895) % 1.0
    return np.asarray(colorsys.hsv_to_rgb(hue, 0.78, 0.95))


def semantic_colors(semantic_id: np.ndarray) -> np.ndarray:
    fallback = np.asarray(SEMANTIC_COLORS[-1], dtype=np.float64)
    colors = np.tile(fallback, (len(semantic_id), 1))
    for value in np.unique(semantic_id):
        color = SEMANTIC_COLORS.get(int(value), SEMANTIC_COLORS[-1])
        colors[semantic_id == value] = color
    return colors


def instance_colors(instance_id: np.ndarray) -> np.ndarray:
    colors = np.empty((len(instance_id), 3), dtype=np.float64)
    for value in np.unique(instance_id):
        colors[instance_id == value] = instance_color(int(value))
    return colors


def dynamic_colors(is_dynamic: np.ndarray) -> np.ndarray:
    colors = np.tile([0.48, 0.48, 0.48], (len(is_dynamic), 1))
    colors[is_dynamic] = [1.0, 0.08, 0.05]
    return colors


def ground_non_ground_colors(semantic_id: np.ndarray) -> np.ndarray:
    """Colour ground, non-ground, undefined and unassigned separately."""
    colors = np.tile(
        GROUND_VIEW_COLORS["nn_unassigned"],
        (len(semantic_id), 1),
    ).astype(np.float64)
    colors[semantic_id == 0] = GROUND_VIEW_COLORS["undefined"]
    non_ground = (semantic_id >= 1) & (semantic_id <= 16)
    colors[non_ground] = GROUND_VIEW_COLORS["non_ground"]
    ground = np.isin(semantic_id, list(GROUND_CLASSES))
    colors[ground] = GROUND_VIEW_COLORS["ground"]
    return colors


class FrameViewer:
    def __init__(self, paths: List[str], args: argparse.Namespace):
        self.paths = paths
        self.args = args
        self.index = 0
        self.color_mode = args.initial_color_mode
        self.show_undefined = not args.hide_undefined
        self.cloud = o3d.geometry.PointCloud()
        self.cache: Dict[str, np.ndarray] = {}

        if args.start_frame is not None:
            numbers = [frame_number(path) for path in paths]
            if args.start_frame not in numbers:
                raise ValueError(
                    f"Frame {args.start_frame} is not present. Available "
                    f"range: {numbers[0]}–{numbers[-1]}"
                )
            self.index = numbers.index(args.start_frame)

    def load_current(self) -> Tuple[np.ndarray, np.ndarray]:
        path = self.paths[self.index]
        with np.load(path) as data:
            required = {"semantic_id", "instance_id", "is_dynamic"}
            missing = required - set(data.files)
            if missing:
                raise KeyError(
                    f"{path} is missing NPZ keys: {sorted(missing)}. "
                    f"Available keys: {data.files}"
                )

            if self.args.coordinate_frame == "vehicle":
                xyz_key = "xyz_vehicle"
                if xyz_key not in data.files:
                    raise KeyError(
                        f"{path} has no xyz_vehicle. Use "
                        "--coordinate-frame world or regenerate the output."
                    )
            else:
                xyz_key = "xyz"

            xyz = data[xyz_key].astype(np.float64)
            semantic_id = data["semantic_id"].astype(np.int32)
            instance_id = data["instance_id"].astype(np.int32)
            is_dynamic = data["is_dynamic"].astype(bool)
            confidence = (
                data["confidence"].astype(np.float32)
                if "confidence" in data.files
                else np.ones(len(xyz), dtype=np.float32)
            )

        if not (
            len(xyz) == len(semantic_id) == len(instance_id)
            == len(is_dynamic) == len(confidence)
        ):
            raise ValueError(f"Array length mismatch in {path}")

        if self.color_mode == "semantic":
            colors = semantic_colors(semantic_id)
        elif self.color_mode == "instance":
            colors = instance_colors(instance_id)
        elif self.color_mode == "dynamic":
            colors = dynamic_colors(is_dynamic)
        else:
            colors = ground_non_ground_colors(semantic_id)

        visible = np.ones(len(xyz), dtype=bool)
        if not self.show_undefined:
            visible &= semantic_id > 0

        self.cache = {
            "semantic_id": semantic_id,
            "instance_id": instance_id,
            "is_dynamic": is_dynamic,
            "confidence": confidence,
            "visible": visible,
        }
        return xyz[visible], colors[visible]

    def print_status(self) -> None:
        path = self.paths[self.index]
        visible = self.cache["visible"]
        semantic_id = self.cache["semantic_id"]
        instance_id = self.cache["instance_id"]
        is_dynamic = self.cache["is_dynamic"]
        unique_semantics = np.unique(semantic_id[visible])
        names = [SEMANTIC_NAMES.get(int(value), str(int(value)))
                 for value in unique_semantics]
        ground_count = np.count_nonzero(
            visible & np.isin(semantic_id, list(GROUND_CLASSES))
        )
        non_ground_count = np.count_nonzero(
            visible & (semantic_id >= 1) & (semantic_id <= 16)
        )
        print(
            f"Frame {frame_number(path):03d} "
            f"[{self.index + 1}/{len(self.paths)}] | "
            f"mode={self.color_mode} | visible={np.count_nonzero(visible):,} "
            f"| dynamic={np.count_nonzero(is_dynamic & visible):,} | "
            f"instances={len(np.unique(instance_id[visible][instance_id[visible] > 0]))} "
            f"| ground={ground_count:,} | non-ground={non_ground_count:,} "
            f"| classes={names}"
        )

    def update(self, vis: o3d.visualization.Visualizer,
               reset_view: bool = False) -> bool:
        xyz, colors = self.load_current()
        self.cloud.points = o3d.utility.Vector3dVector(xyz)
        self.cloud.colors = o3d.utility.Vector3dVector(colors)
        vis.update_geometry(self.cloud)
        if reset_view:
            vis.reset_view_point(True)
        self.print_status()
        return False

    def next_frame(self, vis) -> bool:
        self.index = (self.index + 1) % len(self.paths)
        return self.update(vis)

    def previous_frame(self, vis) -> bool:
        self.index = (self.index - 1) % len(self.paths)
        return self.update(vis)

    def set_semantic(self, vis) -> bool:
        self.color_mode = "semantic"
        return self.update(vis)

    def set_instance(self, vis) -> bool:
        self.color_mode = "instance"
        return self.update(vis)

    def set_dynamic(self, vis) -> bool:
        self.color_mode = "dynamic"
        return self.update(vis)

    def set_ground(self, vis) -> bool:
        self.color_mode = "ground"
        return self.update(vis)

    def toggle_undefined(self, vis) -> bool:
        self.show_undefined = not self.show_undefined
        return self.update(vis)

    def reset_view(self, vis) -> bool:
        return self.update(vis, reset_view=True)

    def screenshot(self, vis) -> bool:
        screenshot_dir = Path(self.args.screenshot_dir)
        screenshot_dir.mkdir(parents=True, exist_ok=True)
        number = frame_number(self.paths[self.index])
        path = screenshot_dir / f"frame_{number:03d}_{self.color_mode}.png"
        vis.capture_screen_image(str(path), do_render=True)
        print(f"Saved screenshot: {path}")
        return False


def main() -> None:
    args = parse_args()
    paths = discover_frames(args.input_dir)
    viewer = FrameViewer(paths, args)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    created = vis.create_window(
        window_name="Waymo semantic + instance viewer",
        width=1600,
        height=900,
    )
    if not created:
        raise RuntimeError(
            "Open3D could not create a window. Run from a graphical desktop "
            "session with a valid DISPLAY, not a headless SSH shell."
        )

    xyz, colors = viewer.load_current()
    viewer.cloud.points = o3d.utility.Vector3dVector(xyz)
    viewer.cloud.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(viewer.cloud)

    render = vis.get_render_option()
    render.background_color = np.asarray([1.0, 1.0, 1.0])
    render.point_size = args.point_size

    # GLFW key codes: right=262, left=263.
    vis.register_key_callback(262, viewer.next_frame)
    vis.register_key_callback(263, viewer.previous_frame)
    vis.register_key_callback(ord("N"), viewer.next_frame)
    vis.register_key_callback(ord("P"), viewer.previous_frame)
    vis.register_key_callback(ord("S"), viewer.set_semantic)
    vis.register_key_callback(ord("I"), viewer.set_instance)
    vis.register_key_callback(ord("D"), viewer.set_dynamic)
    vis.register_key_callback(ord("G"), viewer.set_ground)
    vis.register_key_callback(ord("U"), viewer.toggle_undefined)
    vis.register_key_callback(ord("R"), viewer.reset_view)
    vis.register_key_callback(ord("K"), viewer.screenshot)

    viewer.print_status()
    print(__doc__)
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
