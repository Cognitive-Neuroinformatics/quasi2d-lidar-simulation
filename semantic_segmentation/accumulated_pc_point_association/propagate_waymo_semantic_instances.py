#!/usr/bin/env python3
"""Propagate Waymo TOP-LiDAR semantic and instance labels to every frame.

The script deliberately uses two different propagation mechanisms:

* Static/background points: KNN in the global/world coordinate frame.
* Dynamic foreground points: Waymo 3-D box tracking ID plus KNN in the
  tracked object's local coordinate frame.

A tracked box is only a candidate generator.  Points inside a box are not
automatically assigned to the object.  On segmentation-labelled frames the
script learns positive samples (the object's true instance points) and
negative samples (road/background points that happen to be inside the box).
On missing frames, a candidate must match the positive object reference more
closely than the negative reference.  A robust local ground plane supplies a
fallback/veto, particularly for tracks with few labelled observations.

Output instance IDs are scene-stable integers derived from Waymo laser-label
track IDs.  Zero means background/no instance.  They are intentionally not the
sparse, frame-local instance integers stored in the segmentation range image.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
import yaml
from scipy.spatial import cKDTree
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset import label_pb2
from waymo_open_dataset.utils import frame_utils


# Waymo LiDAR segmentation IDs.  Kept explicit so the saved file remains
# interpretable without importing the Waymo package.
NN_UNASSIGNED = -1
UNDEFINED = 0
CAR = 1
TRUCK = 2
BUS = 3
OTHER_VEHICLE = 4
MOTORCYCLIST = 5
BICYCLIST = 6
PEDESTRIAN = 7
SIGN = 8
TRAFFIC_LIGHT = 9
POLE = 10
CONSTRUCTION_CONE = 11
BICYCLE = 12
MOTORCYCLE = 13
BUILDING = 14
VEGETATION = 15
TREE_TRUNK = 16
CURB = 17
ROAD = 18
LANE_MARKER = 19
OTHER_GROUND = 20
WALKABLE = 21
SIDEWALK = 22

GROUND_SEMANTICS = {CURB, ROAD, LANE_MARKER, OTHER_GROUND, WALKABLE, SIDEWALK}
DYNAMIC_BOX_TYPES = {
    label_pb2.Label.TYPE_VEHICLE,
    label_pb2.Label.TYPE_PEDESTRIAN,
    label_pb2.Label.TYPE_CYCLIST,
}
BOX_TYPE_FALLBACK_SEMANTIC = {
    label_pb2.Label.TYPE_VEHICLE: CAR,
    label_pb2.Label.TYPE_PEDESTRIAN: PEDESTRIAN,
    label_pb2.Label.TYPE_CYCLIST: BICYCLIST,
    label_pb2.Label.TYPE_SIGN: SIGN,
}
BOX_COMPATIBLE_SEMANTICS = {
    label_pb2.Label.TYPE_VEHICLE: {CAR, TRUCK, BUS, OTHER_VEHICLE},
    label_pb2.Label.TYPE_PEDESTRIAN: {PEDESTRIAN},
    label_pb2.Label.TYPE_CYCLIST: {
        MOTORCYCLIST, BICYCLIST, BICYCLE, MOTORCYCLE
    },
    label_pb2.Label.TYPE_SIGN: {SIGN},
}


@dataclass
class BoxState:
    frame_index: int
    timestamp_micros: int
    center_world: np.ndarray
    heading_world: float
    speed: float
    box_type: int


@dataclass
class TrackReference:
    positive_chunks: List[np.ndarray] = field(default_factory=list)
    negative_chunks: List[np.ndarray] = field(default_factory=list)
    semantic_votes: Counter = field(default_factory=Counter)
    positive_tree: Optional[cKDTree] = None
    negative_tree: Optional[cKDTree] = None
    positive_xyz: Optional[np.ndarray] = None
    negative_xyz: Optional[np.ndarray] = None


def flatten_config(mapping: dict, prefix: str = "") -> dict:
    """Flatten readable YAML sections to argparse destination names.

    Section names are organizational only. For example,
    ``static_knn: {static_k: 3}`` becomes ``static_k=3``.
    """
    flattened = {}
    for key, value in mapping.items():
        if isinstance(value, dict):
            nested = flatten_config(value, prefix=f"{prefix}{key}.")
            for nested_key, nested_value in nested.items():
                if nested_key in flattened:
                    raise ValueError(f"Duplicate config key: {nested_key}")
                flattened[nested_key] = nested_value
        else:
            leaf = str(key).replace("-", "_")
            if leaf in flattened:
                raise ValueError(f"Duplicate config key: {prefix}{key}")
            flattened[leaf] = value
    return flattened


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config", type=str, default=None,
        help="YAML configuration file. Command-line options override YAML."
    )
    config_args, _ = config_parser.parse_known_args()

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[config_parser],
    )
    parser.add_argument("--tfrecord", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=-1,
                        help="Inclusive; -1 means the final frame.")

    parser.add_argument("--dynamic-speed-threshold", type=float, default=0.5)
    parser.add_argument("--dynamic-displacement-threshold", type=float,
                        default=1.0)
    parser.add_argument("--min-track-observations", type=int, default=2)

    parser.add_argument("--box-margin-xy", type=float, default=0.05)
    parser.add_argument("--box-margin-z-top", type=float, default=0.05)
    parser.add_argument("--box-margin-z-bottom", type=float, default=0.0)
    parser.add_argument("--association-min-points", type=int, default=3)
    parser.add_argument("--association-min-purity", type=float, default=0.5)

    parser.add_argument("--object-k", type=int, default=3)
    parser.add_argument("--object-max-distance", type=float, default=0.20)
    parser.add_argument("--object-distance-margin", type=float, default=0.03)
    parser.add_argument("--max-reference-points", type=int, default=100000)

    parser.add_argument("--static-k", type=int, default=3)
    parser.add_argument("--static-max-distance", type=float, default=1.4,
                        help="Final validated static-map KNN threshold. This "
                             "is deliberately independent of the much smaller "
                             "dynamic object-space threshold.")
    parser.add_argument("--static-min-vote-fraction", type=float, default=0.5)
    parser.add_argument("--max-static-reference-points", type=int,
                        default=5000000)

    parser.add_argument("--ground-ring-margin", type=float, default=1.0)
    parser.add_argument("--ground-plane-tolerance", type=float, default=0.08)
    parser.add_argument("--ground-ransac-iterations", type=int, default=80)
    parser.add_argument("--ground-min-ring-points", type=int, default=30)
    parser.add_argument("--strong-object-distance", type=float, default=0.08)

    parser.add_argument("--random-seed", type=int, default=13)
    parser.add_argument("--save-combined-map", action="store_true",
                        help="Also concatenate all per-frame results. Moving "
                             "objects will form temporal trails; frame_index "
                             "is retained.")
    parser.add_argument("--overwrite", action="store_true")
    if config_args.config is not None:
        config_path = Path(config_args.config)
        with config_path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, dict):
            parser.error("The YAML root must be a mapping/dictionary.")
        try:
            config_values = flatten_config(loaded)
        except ValueError as error:
            parser.error(str(error))

        valid_destinations = {
            action.dest for action in parser._actions
            if action.dest not in {"help", "config"}
        }
        unknown = sorted(set(config_values) - valid_destinations)
        if unknown:
            parser.error(
                "Unknown YAML configuration keys: " + ", ".join(unknown)
            )
        parser.set_defaults(**config_values)

    args = parser.parse_args()
    if not args.tfrecord:
        parser.error("tfrecord must be provided in YAML or via --tfrecord.")
    if not args.output_dir:
        parser.error("output_dir must be provided in YAML or via --output-dir.")
    return args


def iter_frames(path: str) -> Iterable[Tuple[int, open_dataset.Frame]]:
    dataset = tf.data.TFRecordDataset(path, compression_type="")
    for frame_index, raw in enumerate(dataset):
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(raw.numpy()))
        yield frame_index, frame


def in_requested_range(index: int, args: argparse.Namespace) -> bool:
    return index >= args.start_frame and (
        args.end_frame < 0 or index <= args.end_frame
    )


def pose_matrix(frame: open_dataset.Frame) -> np.ndarray:
    return np.asarray(frame.pose.transform, dtype=np.float64).reshape(4, 4)


def transform_xyz(xyz: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return xyz @ transform[:3, :3].T + transform[:3, 3]


def heading_world_from_vehicle_box(box_heading: float,
                                   vehicle_to_world: np.ndarray) -> float:
    direction_vehicle = np.array([
        math.cos(box_heading), math.sin(box_heading), 0.0
    ])
    direction_world = vehicle_to_world[:3, :3] @ direction_vehicle
    return math.atan2(direction_world[1], direction_world[0])


def unpack_parse_result(frame: open_dataset.Frame):
    """Support common Waymo SDK parse_range_image API variants."""
    parsed = frame_utils.parse_range_image_and_camera_projection(frame)
    if len(parsed) == 4:
        range_images, camera_projections, segmentation_labels, top_pose = parsed
    elif len(parsed) == 3:
        range_images, camera_projections, top_pose = parsed
        segmentation_labels = {}
    else:
        raise RuntimeError(
            "Unexpected parse_range_image_and_camera_projection return count: "
            f"{len(parsed)}"
        )
    return range_images, camera_projections, segmentation_labels, top_pose


def extract_top_both_returns(
    frame: open_dataset.Frame,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Return TOP XYZ, intensity and aligned [instance, semantic] labels.

    Labels are returned only if TOP segmentation labels are present for all
    extracted labelled returns.  TOP return 0 and return 1 are concatenated in
    exactly the same order for points and labels.
    """
    range_images, camera_projections, seg_labels, top_pose = \
        unpack_parse_result(frame)

    calibrations = sorted(frame.context.laser_calibrations,
                          key=lambda c: c.name)
    top_indices = [
        i for i, calibration in enumerate(calibrations)
        if calibration.name == open_dataset.LaserName.TOP
    ]
    if len(top_indices) != 1:
        raise RuntimeError(f"Expected one TOP calibration, found {top_indices}")
    top_index = top_indices[0]

    xyz_parts: List[np.ndarray] = []
    intensity_parts: List[np.ndarray] = []
    label_parts: List[np.ndarray] = []
    labels_complete = True

    for return_index in (0, 1):
        point_clouds, _ = frame_utils.convert_range_image_to_point_cloud(
            frame,
            range_images,
            camera_projections,
            top_pose,
            ri_index=return_index,
        )
        xyz = np.asarray(point_clouds[top_index], dtype=np.float64)

        ri = range_images[open_dataset.LaserName.TOP][return_index]
        ri_tensor = tf.reshape(
            tf.convert_to_tensor(ri.data), ri.shape.dims
        ).numpy()
        valid = ri_tensor[..., 0] > 0
        intensity = np.asarray(ri_tensor[..., 1][valid], dtype=np.float32)

        if len(xyz) != len(intensity):
            raise RuntimeError(
                f"TOP return {return_index}: XYZ/intensity mismatch "
                f"{len(xyz)} != {len(intensity)}"
            )

        xyz_parts.append(xyz)
        intensity_parts.append(intensity)

        available = (
            open_dataset.LaserName.TOP in seg_labels
            and len(seg_labels[open_dataset.LaserName.TOP]) > return_index
        )
        if available:
            label_ri = seg_labels[open_dataset.LaserName.TOP][return_index]
            label_tensor = tf.reshape(
                tf.convert_to_tensor(label_ri.data), label_ri.shape.dims
            ).numpy()
            aligned = np.asarray(label_tensor[valid], dtype=np.int32)
            if aligned.ndim != 2 or aligned.shape[1] < 2:
                raise RuntimeError(
                    f"Unexpected segmentation label shape {aligned.shape}"
                )
            label_parts.append(aligned[:, :2])
        else:
            labels_complete = False

    labels = None
    if labels_complete and len(label_parts) == 2:
        labels = np.concatenate(label_parts, axis=0)

    return (
        np.concatenate(xyz_parts, axis=0),
        np.concatenate(intensity_parts, axis=0),
        labels,
    )


def points_in_box(
    xyz_vehicle: np.ndarray,
    box,
    margin_xy: float = 0.0,
    margin_z_top: float = 0.0,
    margin_z_bottom: float = 0.0,
) -> np.ndarray:
    dx = xyz_vehicle[:, 0] - box.center_x
    dy = xyz_vehicle[:, 1] - box.center_y
    dz = xyz_vehicle[:, 2] - box.center_z
    cosine = math.cos(box.heading)
    sine = math.sin(box.heading)
    local_x = cosine * dx + sine * dy
    local_y = -sine * dx + cosine * dy
    half_l = 0.5 * box.length + margin_xy
    half_w = 0.5 * box.width + margin_xy
    lower_z = -0.5 * box.height - margin_z_bottom
    upper_z = 0.5 * box.height + margin_z_top
    return (
        (np.abs(local_x) <= half_l)
        & (np.abs(local_y) <= half_w)
        & (dz >= lower_z)
        & (dz <= upper_z)
    )


def to_object_coordinates(xyz_vehicle: np.ndarray, box) -> np.ndarray:
    shifted = xyz_vehicle - np.array([
        box.center_x, box.center_y, box.center_z
    ])
    cosine = math.cos(box.heading)
    sine = math.sin(box.heading)
    local = np.empty_like(shifted, dtype=np.float64)
    local[:, 0] = cosine * shifted[:, 0] + sine * shifted[:, 1]
    local[:, 1] = -sine * shifted[:, 0] + cosine * shifted[:, 1]
    local[:, 2] = shifted[:, 2]
    return local


def reservoir_subsample(array: np.ndarray, maximum: int,
                        rng: np.random.Generator) -> np.ndarray:
    if maximum <= 0 or len(array) <= maximum:
        return array
    selected = rng.choice(len(array), size=maximum, replace=False)
    return array[selected]


def query_mean_distance(tree: cKDTree, query: np.ndarray, k: int) -> np.ndarray:
    actual_k = min(k, tree.n)
    distances, _ = tree.query(query, k=actual_k, workers=-1)
    if actual_k == 1:
        return np.asarray(distances, dtype=np.float64)
    return np.asarray(distances, dtype=np.float64).mean(axis=1)


def local_ground_ring(xyz_vehicle: np.ndarray, box,
                      ring_margin: float) -> np.ndarray:
    outer = points_in_box(
        xyz_vehicle,
        box,
        margin_xy=ring_margin,
        margin_z_top=0.0,
        margin_z_bottom=0.5,
    )
    inner_xy = points_in_box(
        xyz_vehicle,
        box,
        margin_xy=0.15,
        margin_z_top=0.0,
        margin_z_bottom=0.5,
    )
    ring = outer & ~inner_xy
    # Restrict the fitting data to the vertical neighbourhood of the box base.
    expected_base = box.center_z - 0.5 * box.height
    ring &= np.abs(xyz_vehicle[:, 2] - expected_base) < 0.45
    return xyz_vehicle[ring]


def fit_ground_plane_ransac(
    points: np.ndarray,
    iterations: int,
    rng: np.random.Generator,
) -> Optional[np.ndarray]:
    if len(points) < 3:
        return None
    best_plane = None
    best_count = 0
    for _ in range(iterations):
        sample = points[rng.choice(len(points), size=3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-8:
            continue
        normal /= norm
        if normal[2] < 0:
            normal = -normal
        # Roads are locally near-horizontal.  Reject walls/vehicle sides.
        if normal[2] < math.cos(math.radians(25.0)):
            continue
        d = -float(normal @ sample[0])
        distances = np.abs(points @ normal + d)
        count = int(np.count_nonzero(distances < 0.06))
        if count > best_count:
            best_count = count
            best_plane = np.r_[normal, d]
    return best_plane


def ground_distance(xyz: np.ndarray, plane: Optional[np.ndarray]) -> np.ndarray:
    if plane is None:
        return np.full(len(xyz), np.inf, dtype=np.float64)
    return np.abs(xyz @ plane[:3] + plane[3])


def box_labels_for_dynamic_tracks(frame, dynamic_tracks: set):
    return [
        label for label in frame.laser_labels
        if label.id in dynamic_tracks and label.type in DYNAMIC_BOX_TYPES
    ]


def collect_tracks(args: argparse.Namespace) -> Dict[str, List[BoxState]]:
    tracks: Dict[str, List[BoxState]] = defaultdict(list)
    for frame_index, frame in iter_frames(args.tfrecord):
        if not in_requested_range(frame_index, args):
            continue
        vehicle_to_world = pose_matrix(frame)
        for label in frame.laser_labels:
            if label.type not in DYNAMIC_BOX_TYPES:
                continue
            center_vehicle = np.array([[
                label.box.center_x,
                label.box.center_y,
                label.box.center_z,
            ]])
            center_world = transform_xyz(center_vehicle, vehicle_to_world)[0]
            speed = math.hypot(
                float(label.metadata.speed_x),
                float(label.metadata.speed_y),
            )
            tracks[label.id].append(BoxState(
                frame_index=frame_index,
                timestamp_micros=int(frame.timestamp_micros),
                center_world=center_world,
                heading_world=heading_world_from_vehicle_box(
                    label.box.heading, vehicle_to_world
                ),
                speed=speed,
                box_type=int(label.type),
            ))
    return tracks


def select_dynamic_tracks(
    tracks: Dict[str, List[BoxState]], args: argparse.Namespace
) -> Tuple[set, Dict[str, dict]]:
    selected = set()
    summaries = {}
    for track_id, states in tracks.items():
        centers = np.stack([state.center_world for state in states])
        displacement = float(np.max(
            np.linalg.norm(centers - centers[0], axis=1)
        ))
        speeds = np.asarray([state.speed for state in states])
        median_speed = float(np.median(speeds))
        dynamic = (
            len(states) >= args.min_track_observations
            and (
                displacement >= args.dynamic_displacement_threshold
                or median_speed >= args.dynamic_speed_threshold
            )
        )
        if dynamic:
            selected.add(track_id)
        summaries[track_id] = {
            "box_type": int(states[0].box_type),
            "observations": len(states),
            "first_frame": int(states[0].frame_index),
            "last_frame": int(states[-1].frame_index),
            "displacement_m": displacement,
            "median_speed_mps": median_speed,
            "is_dynamic": dynamic,
        }
    return selected, summaries


def majority_nonzero(values: np.ndarray) -> Tuple[int, int, float]:
    values = values[values > 0]
    if len(values) == 0:
        return 0, 0, 0.0
    labels, counts = np.unique(values, return_counts=True)
    winner = int(labels[np.argmax(counts)])
    count = int(np.max(counts))
    return winner, count, count / len(values)


def build_references(
    args: argparse.Namespace,
    dynamic_tracks: set,
    rng: np.random.Generator,
) -> Tuple[Dict[str, TrackReference], np.ndarray, np.ndarray, List[int]]:
    references: Dict[str, TrackReference] = defaultdict(TrackReference)
    static_xyz_chunks: List[np.ndarray] = []
    static_semantic_chunks: List[np.ndarray] = []
    labelled_frames: List[int] = []

    for frame_index, frame in iter_frames(args.tfrecord):
        if not in_requested_range(frame_index, args):
            continue
        xyz_vehicle, _, seg = extract_top_both_returns(frame)
        if seg is None:
            continue
        labelled_frames.append(frame_index)
        gt_instance = seg[:, 0]
        gt_semantic = seg[:, 1]
        dynamic_union = np.zeros(len(xyz_vehicle), dtype=bool)

        for label in box_labels_for_dynamic_tracks(frame, dynamic_tracks):
            candidate = points_in_box(
                xyz_vehicle,
                label.box,
                args.box_margin_xy,
                args.box_margin_z_top,
                args.box_margin_z_bottom,
            )
            dynamic_union |= candidate
            candidate_idx = np.flatnonzero(candidate)
            if len(candidate_idx) == 0:
                continue

            compatible = BOX_COMPATIBLE_SEMANTICS.get(label.type, set())
            eligible = candidate & np.isin(gt_semantic, list(compatible))
            instance_value, count, purity = majority_nonzero(
                gt_instance[eligible]
            )
            if (
                instance_value == 0
                or count < args.association_min_points
                or purity < args.association_min_purity
            ):
                continue

            positive = candidate & (gt_instance == instance_value)
            # Negative examples include ground/background inside this box but
            # exclude pixels belonging to a different foreground instance;
            # those are usually box-overlap cases rather than useful negatives.
            negative = candidate & (gt_instance == 0)

            reference = references[label.id]
            if np.any(positive):
                reference.positive_chunks.append(
                    to_object_coordinates(xyz_vehicle[positive], label.box)
                )
                positive_semantics = gt_semantic[positive]
                for semantic in positive_semantics:
                    if int(semantic) in compatible:
                        reference.semantic_votes[int(semantic)] += 1
            if np.any(negative):
                reference.negative_chunks.append(
                    to_object_coordinates(xyz_vehicle[negative], label.box)
                )

        static_mask = (~dynamic_union) & (gt_semantic != UNDEFINED)
        if np.any(static_mask):
            xyz_world = transform_xyz(
                xyz_vehicle[static_mask], pose_matrix(frame)
            )
            static_xyz_chunks.append(xyz_world.astype(np.float32))
            static_semantic_chunks.append(
                gt_semantic[static_mask].astype(np.int16)
            )

        print(
            f"Pass 2 frame {frame_index:03d}: labelled, "
            f"static references={int(np.count_nonzero(static_mask))}"
        )

    if not labelled_frames:
        raise RuntimeError("No TOP segmentation-labelled frames were detected.")

    for track_id, reference in references.items():
        if reference.positive_chunks:
            positive = np.concatenate(reference.positive_chunks, axis=0)
            positive = reservoir_subsample(
                positive, args.max_reference_points, rng
            )
            reference.positive_xyz = positive
            reference.positive_tree = cKDTree(positive)
        if reference.negative_chunks:
            negative = np.concatenate(reference.negative_chunks, axis=0)
            negative = reservoir_subsample(
                negative, args.max_reference_points, rng
            )
            reference.negative_xyz = negative
            reference.negative_tree = cKDTree(negative)
        reference.positive_chunks.clear()
        reference.negative_chunks.clear()

    static_xyz = np.concatenate(static_xyz_chunks, axis=0)
    static_semantic = np.concatenate(static_semantic_chunks, axis=0)
    if len(static_xyz) > args.max_static_reference_points > 0:
        chosen = rng.choice(
            len(static_xyz),
            size=args.max_static_reference_points,
            replace=False,
        )
        static_xyz = static_xyz[chosen]
        static_semantic = static_semantic[chosen]

    return references, static_xyz, static_semantic, labelled_frames


def track_semantic(
    track_id: str,
    box_type: int,
    references: Dict[str, TrackReference],
) -> int:
    reference = references.get(track_id)
    if reference is not None and reference.semantic_votes:
        return int(reference.semantic_votes.most_common(1)[0][0])
    return int(BOX_TYPE_FALLBACK_SEMANTIC.get(box_type, UNDEFINED))


def classify_object_candidates(
    xyz_vehicle: np.ndarray,
    candidate_idx: np.ndarray,
    label,
    reference: Optional[TrackReference],
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return accepted candidate indices and confidence scores."""
    if len(candidate_idx) == 0:
        return candidate_idx, np.empty(0, dtype=np.float32)

    candidate_xyz = xyz_vehicle[candidate_idx]
    object_xyz = to_object_coordinates(candidate_xyz, label.box)

    positive_distance = np.full(len(candidate_idx), np.inf)
    negative_distance = np.full(len(candidate_idx), np.inf)
    if reference is not None and reference.positive_tree is not None:
        positive_distance = query_mean_distance(
            reference.positive_tree, object_xyz, args.object_k
        )
    if reference is not None and reference.negative_tree is not None:
        negative_distance = query_mean_distance(
            reference.negative_tree, object_xyz, args.object_k
        )

    has_positive_model = np.isfinite(positive_distance).any()
    if has_positive_model:
        accepted = positive_distance <= args.object_max_distance
        if np.isfinite(negative_distance).any():
            accepted &= (
                positive_distance + args.object_distance_margin
                < negative_distance
            )
    else:
        # No segmentation anchor ever observed this track. Start with box
        # candidates, then conservatively remove the local road plane below.
        accepted = np.ones(len(candidate_idx), dtype=bool)

    ring = local_ground_ring(
        xyz_vehicle, label.box, args.ground_ring_margin
    )
    plane = None
    if len(ring) >= args.ground_min_ring_points:
        plane = fit_ground_plane_ransac(
            ring, args.ground_ransac_iterations, rng
        )
    near_ground = (
        ground_distance(candidate_xyz, plane)
        <= args.ground_plane_tolerance
    )
    strong_object_match = positive_distance <= args.strong_object_distance
    accepted &= ~(near_ground & ~strong_object_match)

    # Larger confidence is better. It is used only to resolve overlapping boxes.
    if has_positive_model:
        confidence = (
            args.object_max_distance - positive_distance
        ).astype(np.float32)
        finite_negative = np.isfinite(negative_distance)
        confidence[finite_negative] += np.maximum(
            0.0,
            negative_distance[finite_negative]
            - positive_distance[finite_negative],
        ).astype(np.float32)
    else:
        confidence = np.full(len(candidate_idx), 0.001, dtype=np.float32)
    confidence[near_ground] -= 0.5
    return candidate_idx[accepted], confidence[accepted]


def propagate_static_semantics(
    query_world: np.ndarray,
    tree: cKDTree,
    reference_semantic: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(query_world) == 0:
        return np.empty(0, dtype=np.int16), np.empty(0, dtype=np.float32)
    k = min(args.static_k, tree.n)
    distances, indices = tree.query(query_world, k=k, workers=-1)
    if k == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    neighbour_labels = reference_semantic[indices]
    within = distances <= args.static_max_distance
    # There are only 23 Waymo semantic classes. Counting per class is much
    # faster than a Python loop over ~150k points per frame.
    class_counts = np.zeros((len(query_world), 23), dtype=np.uint8)
    rows = np.arange(len(query_world))
    for neighbour in range(k):
        valid_rows = rows[within[:, neighbour]]
        valid_classes = neighbour_labels[
            within[:, neighbour], neighbour
        ].astype(np.int64)
        np.add.at(class_counts, (valid_rows, valid_classes), 1)
    result = np.argmax(class_counts, axis=1).astype(np.int16)
    winning_votes = np.max(class_counts, axis=1)
    valid_neighbour_count = np.sum(within, axis=1)
    confidence = np.divide(
        winning_votes,
        valid_neighbour_count,
        out=np.zeros(len(query_world), dtype=np.float32),
        where=valid_neighbour_count > 0,
    )
    accepted = (
        (valid_neighbour_count > 0)
        & (confidence >= args.static_min_vote_fraction)
    )
    result[~accepted] = NN_UNASSIGNED
    confidence[~accepted] = 0.0
    return result, confidence


def save_json(path: Path, payload) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def process_and_save(
    args: argparse.Namespace,
    dynamic_tracks: set,
    track_summaries: Dict[str, dict],
    references: Dict[str, TrackReference],
    static_xyz: np.ndarray,
    static_semantic: np.ndarray,
    labelled_frames: Sequence[int],
    rng: np.random.Generator,
) -> None:
    output_dir = Path(args.output_dir)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    track_to_instance = {
        track_id: index + 1
        for index, track_id in enumerate(sorted(dynamic_tracks))
    }
    track_to_semantic = {
        track_id: track_semantic(
            track_id,
            track_summaries[track_id]["box_type"],
            references,
        )
        for track_id in sorted(dynamic_tracks)
    }

    static_tree = cKDTree(static_xyz)
    labelled_set = set(labelled_frames)
    combined_chunks = defaultdict(list)

    for frame_index, frame in iter_frames(args.tfrecord):
        if not in_requested_range(frame_index, args):
            continue
        output_path = frames_dir / f"frame_{frame_index:03d}.npz"
        if output_path.exists() and not args.overwrite:
            print(f"Pass 3 frame {frame_index:03d}: exists, skipped")
            continue

        xyz_vehicle, intensity, gt_seg = extract_top_both_returns(frame)
        vehicle_to_world = pose_matrix(frame)
        xyz_world = transform_xyz(xyz_vehicle, vehicle_to_world)
        point_count = len(xyz_vehicle)

        # Keep propagation failure (-1) distinct from Waymo's genuine
        # UNDEFINED semantic class (0).
        semantic_id = np.full(
            point_count, NN_UNASSIGNED, dtype=np.int16
        )
        instance_id = np.zeros(point_count, dtype=np.int32)
        is_dynamic = np.zeros(point_count, dtype=bool)
        confidence = np.zeros(point_count, dtype=np.float32)

        # Dynamic assignment happens first. A box only produces candidates;
        # positive/negative object references and the ground plane decide which
        # candidates receive the track's stable instance ID.
        for label in box_labels_for_dynamic_tracks(frame, dynamic_tracks):
            candidate = points_in_box(
                xyz_vehicle,
                label.box,
                args.box_margin_xy,
                args.box_margin_z_top,
                args.box_margin_z_bottom,
            )
            candidate_idx = np.flatnonzero(candidate)
            accepted_idx = np.empty(0, dtype=np.int64)
            accepted_confidence = np.empty(0, dtype=np.float32)

            # On a genuinely labelled frame, use the exact segmentation mask.
            # The box is used only to associate its persistent track ID with
            # the frame's sparse/raw instance integer.
            if gt_seg is not None and len(candidate_idx) > 0:
                gt_instance = gt_seg[:, 0]
                gt_semantic = gt_seg[:, 1]
                compatible = BOX_COMPATIBLE_SEMANTICS.get(label.type, set())
                eligible = candidate & np.isin(
                    gt_semantic, list(compatible)
                )
                raw_instance, count, purity = majority_nonzero(
                    gt_instance[eligible]
                )
                if (
                    raw_instance > 0
                    and count >= args.association_min_points
                    and purity >= args.association_min_purity
                ):
                    accepted_idx = np.flatnonzero(
                        candidate & (gt_instance == raw_instance)
                    )
                    accepted_confidence = np.ones(
                        len(accepted_idx), dtype=np.float32
                    )

            # Missing segmentation frame, or an anchor whose box/instance
            # association was too weak: use the learned object-space model.
            if len(accepted_idx) == 0:
                accepted_idx, accepted_confidence = \
                    classify_object_candidates(
                        xyz_vehicle,
                        candidate_idx,
                        label,
                        references.get(label.id),
                        args,
                        rng,
                    )
            replace = accepted_confidence > confidence[accepted_idx]
            accepted_idx = accepted_idx[replace]
            accepted_confidence = accepted_confidence[replace]
            instance_id[accepted_idx] = track_to_instance[label.id]
            semantic_id[accepted_idx] = track_to_semantic[label.id]
            is_dynamic[accepted_idx] = True
            confidence[accepted_idx] = accepted_confidence

        # Static/background points use exact ground truth on labelled frames;
        # otherwise use the global static reference KNN.
        static_mask = ~is_dynamic
        if gt_seg is not None:
            semantic_id[static_mask] = gt_seg[static_mask, 1].astype(np.int16)
            confidence[static_mask] = (
                semantic_id[static_mask] != UNDEFINED
            ).astype(np.float32)
        else:
            propagated, static_confidence = propagate_static_semantics(
                xyz_world[static_mask],
                static_tree,
                static_semantic,
                args,
            )
            semantic_id[static_mask] = propagated
            confidence[static_mask] = static_confidence

        np.savez_compressed(
            output_path,
            xyz=xyz_world.astype(np.float32),
            xyz_vehicle=xyz_vehicle.astype(np.float32),
            intensity=intensity.astype(np.float32),
            semantic_id=semantic_id,
            instance_id=instance_id,
            is_dynamic=is_dynamic,
            confidence=confidence,
            frame_index=np.full(point_count, frame_index, dtype=np.int16),
            timestamp_micros=np.asarray(
                [frame.timestamp_micros], dtype=np.int64
            ),
            vehicle_to_world=vehicle_to_world.astype(np.float64),
            has_ground_truth_segmentation=np.asarray(
                [frame_index in labelled_set], dtype=bool
            ),
        )

        if args.save_combined_map:
            combined_chunks["xyz"].append(xyz_world.astype(np.float32))
            combined_chunks["intensity"].append(intensity.astype(np.float32))
            combined_chunks["semantic_id"].append(semantic_id)
            combined_chunks["instance_id"].append(instance_id)
            combined_chunks["is_dynamic"].append(is_dynamic)
            combined_chunks["confidence"].append(confidence)
            combined_chunks["frame_index"].append(
                np.full(point_count, frame_index, dtype=np.int16)
            )

        print(
            f"Pass 3 frame {frame_index:03d}: points={point_count:,}, "
            f"dynamic={np.count_nonzero(is_dynamic):,}, "
            f"NN-unassigned="
            f"{np.count_nonzero(semantic_id == NN_UNASSIGNED):,}, "
            f"Waymo-undefined="
            f"{np.count_nonzero(semantic_id == UNDEFINED):,}"
        )

    if args.save_combined_map and combined_chunks:
        np.savez_compressed(
            output_dir / "combined_accumulated_map.npz",
            **{
                key: np.concatenate(chunks, axis=0)
                for key, chunks in combined_chunks.items()
            },
        )

    mapping = {
        "background_instance_id": 0,
        "tracks": {
            str(track_to_instance[track_id]): {
                "waymo_track_id": track_id,
                "semantic_id": int(track_to_semantic[track_id]),
                **track_summaries[track_id],
                "has_positive_reference": bool(
                    track_id in references
                    and references[track_id].positive_tree is not None
                ),
            }
            for track_id in sorted(dynamic_tracks)
        },
    }
    save_json(output_dir / "track_instance_mapping.json", mapping)
    save_json(output_dir / "run_configuration.json", vars(args))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.random_seed)

    print("Pass 1/3: collecting and classifying Waymo box tracks")
    tracks = collect_tracks(args)
    dynamic_tracks, track_summaries = select_dynamic_tracks(tracks, args)
    print(
        f"Found {len(tracks)} vehicle/pedestrian/cyclist tracks; "
        f"selected {len(dynamic_tracks)} dynamic tracks."
    )

    print("Pass 2/3: building labelled static and object references")
    references, static_xyz, static_semantic, labelled_frames = build_references(
        args, dynamic_tracks, rng
    )
    print(f"Detected labelled frames dynamically: {labelled_frames}")
    print(f"Static reference points: {len(static_xyz):,}")
    print(
        "Dynamic tracks with positive segmentation references: "
        f"{sum(r.positive_tree is not None for r in references.values())}"
    )

    print("Pass 3/3: propagating labels and saving every frame")
    process_and_save(
        args,
        dynamic_tracks,
        track_summaries,
        references,
        static_xyz,
        static_semantic,
        labelled_frames,
        rng,
    )
    print(f"Finished. Output written to: {output_dir}")


if __name__ == "__main__":
    main()
