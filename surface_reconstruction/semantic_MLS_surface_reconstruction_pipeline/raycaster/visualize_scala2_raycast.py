#!/usr/bin/env python3
"""Interactive SCALA2 MLS raycast viewer with instance-aware 3D boxes."""

from __future__ import annotations

import argparse
import colorsys
import json
from pathlib import Path

import numpy as np
from visualize_mls_surface import SEMANTIC_COLORS

BOX_EDGES = np.asarray([[0,1],[1,3],[3,2],[2,0],[4,5],[5,7],[7,6],[6,4],[0,4],[1,5],[2,6],[3,7]], dtype=np.int32)
STATIC_BOX_COLOR = np.asarray([0.05,0.20,0.90], dtype=np.float64)
DYNAMIC_BOX_COLOR = np.asarray([0.90,0.08,0.08], dtype=np.float64)


def require_open3d():
    try:
        import open3d as o3d
    except ImportError as error:
        raise RuntimeError("Install open3d to use this viewer") from error
    return o3d


def instance_color(value: int):
    return [0.35,0.35,0.35] if value <= 0 else colorsys.hsv_to_rgb((value * 0.618033988749895) % 1.0, 0.8, 0.95)


def point_colors(data, mode):
    count = len(data["xyz"])
    if mode == "uniform":
        return np.tile([0.05,0.35,0.95], (count,1))
    if mode == "semantic":
        return np.asarray([SEMANTIC_COLORS.get(int(v), [0.15,0.15,0.15]) for v in data["semantic_id"]], dtype=np.float64)
    if mode == "ground":
        palette = {-1:[0.2,0.2,0.2], 0:[0.95,0.25,0.15], 1:[0.1,0.75,0.2]}
        return np.asarray([palette.get(int(v), [0.2,0.2,0.2]) for v in data["ground_id"]], dtype=np.float64)
    if mode == "instance":
        return np.asarray([instance_color(int(v)) for v in data["instance_id"]], dtype=np.float64)
    if mode == "source":
        palette = {0:[0.10,0.35,0.95], 1:[1.00,0.20,0.05]}
        return np.asarray([palette.get(int(v), [0.2,0.2,0.2]) for v in data["source_type"]], dtype=np.float64)
    raise ValueError(f"Unknown point-color mode: {mode}")


def transform_points(points, transform):
    points = np.asarray(points, dtype=np.float64)
    transform = np.asarray(transform, dtype=np.float64).reshape(4,4)
    return (np.column_stack([points, np.ones(len(points), dtype=np.float64)]) @ transform.T)[:,:3]


def box_local_corners(length, width, height):
    x, y, z = 0.5 * float(length), 0.5 * float(width), 0.5 * float(height)
    return np.asarray([[-x,-y,-z],[-x,+y,-z],[+x,-y,-z],[+x,+y,-z],[-x,-y,+z],[-x,+y,+z],[+x,-y,+z],[+x,+y,+z]], dtype=np.float64)


def make_box_lineset(o3d, corners_sensor, color):
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.asarray(corners_sensor, dtype=np.float64))
    line_set.lines = o3d.utility.Vector2iVector(BOX_EDGES)
    line_set.colors = o3d.utility.Vector3dVector(np.tile(np.asarray(color, dtype=np.float64), (len(BOX_EDGES),1)))
    return line_set


def infer_caseid(input_dir):
    for path in [input_dir, *input_dir.parents]:
        if path.name.startswith("segment-"):
            return path.name
    return None


def discover_tracks_json(input_dir, explicit_path=None, dataset_root=None, caseid=None):
    if explicit_path is not None:
        path = explicit_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Track file does not exist: {path}")
        return path

    caseid = caseid or infer_caseid(input_dir)
    if caseid is None:
        return None

    if dataset_root is not None:
        candidate = dataset_root.expanduser().resolve() / "temp" / caseid / "stage_a_tracks.json"
        return candidate if candidate.is_file() else None

    for ancestor in [input_dir, *input_dir.parents]:
        candidate = ancestor / "temp" / caseid / "stage_a_tracks.json"
        if candidate.is_file():
            return candidate
    return None


def normalize_tracks(payload):
    raw_tracks = payload.get("tracks", payload)
    if isinstance(raw_tracks, dict):
        iterable = raw_tracks.items()
    elif isinstance(raw_tracks, list):
        iterable = enumerate(raw_tracks)
    else:
        raise ValueError("Unsupported stage_a_tracks.json structure")

    by_instance = {}
    for key, track in iterable:
        if not isinstance(track, dict):
            continue
        instance_id = track.get("instance_id")
        if instance_id is None:
            try:
                instance_id = int(key)
            except (TypeError, ValueError):
                continue
        by_instance[int(instance_id)] = track
    return by_instance


def load_tracks(path):
    if path is None:
        return {}
    with path.open() as stream:
        payload = json.load(stream)
    tracks = normalize_tracks(payload)
    print(f"Loaded {len(tracks):,} tracks from {path}")
    return tracks


def frame_indices_from_npz(data, path):
    output_index = int(np.asarray(data["output_frame_index"]).reshape(-1)[0])
    source_index = int(np.asarray(data["source_frame_index"]).reshape(-1)[0])
    try:
        file_index = int(path.stem)
    except ValueError:
        file_index = output_index
    return output_index, source_index, file_index


def find_frame_record(track, output_index, source_index, file_index):
    frames = track.get("frames", {})
    if not isinstance(frames, dict):
        return None, None

    seen = set()
    for label, value in [("output_frame_index",output_index),("source_frame_index",source_index),("file_index",file_index)]:
        if value in seen:
            continue
        seen.add(value)
        for key in (str(value), value):
            if key in frames:
                return frames[key], label
    return None, None


def yaw_pose_vehicle(box_vehicle):
    box_vehicle = np.asarray(box_vehicle, dtype=np.float64).reshape(-1)
    if box_vehicle.size < 7:
        return None

    cx, cy, cz = box_vehicle[:3]
    heading = float(box_vehicle[6])
    c, s = np.cos(heading), np.sin(heading)
    pose = np.eye(4, dtype=np.float64)
    pose[:3,:3] = np.asarray([[c,-s,0.0],[s,c,0.0],[0.0,0.0,1.0]])
    pose[:3,3] = [cx,cy,cz]
    return pose


def extract_box_geometry(frame_record, data):
    if not isinstance(frame_record, dict):
        return None, None

    box_vehicle = np.asarray(frame_record.get("box_vehicle", []), dtype=np.float64).reshape(-1)
    if box_vehicle.size < 6:
        return None, None

    dimensions = box_vehicle[3:6].astype(np.float64)
    pose_world = np.asarray(frame_record.get("box_pose_world", []), dtype=np.float64)
    if pose_world.size == 16:
        return dimensions, pose_world.reshape(4,4)

    box_to_vehicle = yaw_pose_vehicle(box_vehicle)
    if box_to_vehicle is None:
        return None, None

    vehicle_to_world = np.asarray(data["vehicle_to_world"], dtype=np.float64).reshape(4,4)
    return dimensions, vehicle_to_world @ box_to_vehicle


def rendered_instance_statistics(data):
    instance_id = np.asarray(data["instance_id"], dtype=np.int64)
    source_type = np.asarray(data["source_type"], dtype=np.int64)
    stats = {}

    for instance in np.unique(instance_id):
        instance = int(instance)
        if instance <= 0:
            continue

        mask = instance_id == instance
        source_values, source_counts = np.unique(source_type[mask], return_counts=True)
        stats[instance] = {
            "points": int(np.count_nonzero(mask)),
            "source_type": int(source_values[int(np.argmax(source_counts))]),
            "source_counts": {int(v):int(c) for v,c in zip(source_values, source_counts)}
        }
    return stats


def box_color_from_source_type(source_type):
    return DYNAMIC_BOX_COLOR if int(source_type) == 1 else STATIC_BOX_COLOR


def create_boxes_for_frame(o3d, tracks_by_instance, data, path, min_box_points):
    output_index, source_index, file_index = frame_indices_from_npz(data, path)
    world_to_sensor = np.asarray(data["world_to_sensor"], dtype=np.float64).reshape(4,4)
    instance_stats = rendered_instance_statistics(data)

    geometries, records = [], []
    for instance_id, stats in sorted(instance_stats.items()):
        base = {"instance_id":instance_id, "points":stats["points"], "source_type":stats["source_type"], "source_counts":stats["source_counts"]}

        if stats["points"] < min_box_points:
            records.append({**base, "status":"below_threshold"})
            continue

        track = tracks_by_instance.get(instance_id)
        if track is None:
            records.append({**base, "status":"track_not_found"})
            continue

        frame_record, frame_key_used = find_frame_record(track, output_index, source_index, file_index)
        if frame_record is None:
            records.append({**base, "status":"frame_box_not_found"})
            continue

        dimensions, box_to_world = extract_box_geometry(frame_record, data)
        if dimensions is None or box_to_world is None:
            records.append({**base, "status":"invalid_box_geometry"})
            continue

        length, width, height = dimensions
        corners_world = transform_points(box_local_corners(length, width, height), box_to_world)
        corners_sensor = transform_points(corners_world, world_to_sensor)
        geometries.append(make_box_lineset(o3d, corners_sensor, box_color_from_source_type(stats["source_type"])))
        records.append({**base, "status":"drawn", "frame_key_used":frame_key_used})

    return geometries, records


def print_box_summary(records):
    if not records:
        print("  boxes: no rendered instance_id > 0")
        return

    drawn = [r for r in records if r["status"] == "drawn"]
    skipped = [r for r in records if r["status"] != "drawn"]
    static_drawn = sum(r["source_type"] == 0 for r in drawn)
    dynamic_drawn = sum(r["source_type"] == 1 for r in drawn)
    print(f"  boxes drawn={len(drawn)} (static-source blue={static_drawn}, dynamic-source red={dynamic_drawn})")

    for r in skipped:
        print(f"    skip instance={r['instance_id']} points={r['points']} source={r['source_type']} reason={r['status']}")


def load_frame(path):
    required = ("xyz","semantic_id","ground_id","instance_id","source_type","source_object_id","mirror_side","range","world_to_sensor","vehicle_to_world","output_frame_index","source_frame_index")
    with np.load(path, allow_pickle=False) as loaded:
        missing = [name for name in required if name not in loaded.files]
        if missing:
            raise KeyError(f"{path} is missing required fields: {missing}")
        return {name:np.asarray(loaded[name]) for name in required}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path, help="Directory containing 000.npz, 001.npz, ...")
    parser.add_argument("--point-size", type=float, default=3.0)
    parser.add_argument("--min-box-points", type=int, default=4, help="Minimum rendered points for an instance box. Default: 4.")
    parser.add_argument("--tracks-json", type=Path, default=None, help="Explicit path to stage_a_tracks.json.")
    parser.add_argument("--dataset-root", type=Path, default=None, help="Optional root containing temp/<caseid>/stage_a_tracks.json.")
    parser.add_argument("--caseid", default=None, help="Case ID; normally inferred from --input-dir.")
    parser.add_argument("--no-boxes", action="store_true", help="Start with boxes disabled.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.min_box_points < 0:
        raise ValueError("--min-box-points must be >= 0")

    o3d = require_open3d()
    input_dir = args.input_dir.expanduser().resolve()
    files = sorted(input_dir.glob("*.npz"), key=lambda path: int(path.stem))
    if not files:
        raise FileNotFoundError(f"No NPZ frames in {input_dir}")

    tracks_path = discover_tracks_json(input_dir, explicit_path=args.tracks_json, dataset_root=args.dataset_root, caseid=args.caseid)
    tracks_by_instance = load_tracks(tracks_path) if tracks_path is not None else {}
    if tracks_path is None:
        print("WARNING: stage_a_tracks.json was not found. Point-cloud viewing works, but boxes cannot be drawn.")

    state = {"index":0, "mode":"semantic", "boxes_visible":not args.no_boxes and tracks_path is not None}
    cloud = o3d.geometry.PointCloud()
    active_box_geometries = []

    def remove_current_boxes(vis):
        while active_box_geometries:
            vis.remove_geometry(active_box_geometries.pop(), reset_bounding_box=False)

    def update(vis, reset=False):
        path = files[state["index"]]
        data = load_frame(path)

        cloud.points = o3d.utility.Vector3dVector(data["xyz"].astype(np.float64))
        cloud.colors = o3d.utility.Vector3dVector(point_colors(data, state["mode"]).astype(np.float64))
        vis.update_geometry(cloud)

        remove_current_boxes(vis)
        box_records = []

        if state["boxes_visible"] and tracks_by_instance:
            new_boxes, box_records = create_boxes_for_frame(o3d, tracks_by_instance, data, path, args.min_box_points)
            for geometry in new_boxes:
                vis.add_geometry(geometry, reset_bounding_box=False)
                active_box_geometries.append(geometry)

        if reset:
            vis.reset_view_point(True)

        mirror = int(data["mirror_side"][0]) if len(data["mirror_side"]) else -1
        min_range = float(np.nanmin(data["range"])) if len(data["range"]) else float("nan")
        max_range = float(np.nanmax(data["range"])) if len(data["range"]) else float("nan")
        output_index, source_index, _ = frame_indices_from_npz(data, path)

        print(f"\nFrame {path.stem} ({state['index']+1}/{len(files)}), output_frame={output_index}, source_frame={source_index}, MS{mirror}, view={state['mode']}, hits={len(data['xyz']):,}, range={min_range:.2f}..{max_range:.2f} m, boxes={'ON' if state['boxes_visible'] else 'OFF'}")
        if state["boxes_visible"]:
            print_box_summary(box_records)
        print("  N/Right next | P/Left previous | S semantic | G ground | I instance | O source | U uniform | B boxes on/off")
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

    def toggle_boxes(vis):
        state["boxes_visible"] = not state["boxes_visible"]
        if state["boxes_visible"] and not tracks_by_instance:
            print("Cannot enable boxes: stage_a_tracks.json is unavailable.")
            state["boxes_visible"] = False
        return update(vis)

    visualizer = o3d.visualization.VisualizerWithKeyCallback()
    visualizer.create_window(window_name="SCALA2 MLS raycast + instance-aware boxes", width=1400, height=900)
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
    visualizer.register_key_callback(ord("B"), toggle_boxes)

    options = visualizer.get_render_option()
    options.background_color = np.asarray([1.0,1.0,1.0])
    options.point_size = args.point_size

    update(visualizer, reset=True)
    visualizer.run()
    visualizer.destroy_window()


if __name__ == "__main__":
    main()
