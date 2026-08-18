#!/usr/bin/env python3

import argparse
import json
import os

import numpy as np
import tensorflow as tf

from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import frame_utils


def load_dynamic_track_ids(lidargs_root, case):
    """
    Load whole-track dynamic/static classification created by waymo_preprocessing.py.
    """
    path = os.path.join(
        lidargs_root,
        "temp",
        case,
        "track_classification.json",
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            "track_classification.json was not found:\n"
            f"{path}\n"
            "Run waymo_preprocessing.py first."
        )

    with open(path, "r") as f:
        report = json.load(f)

    dynamic_ids = {
        track_id
        for track_id, info in report.items()
        if bool(info.get("dynamic", False))
    }

    return dynamic_ids, path


def box_mask(points_vehicle, box, margin=0.0):
    """
    Return mask for points geometrically inside one Waymo 3D bounding box.
    """
    dx = points_vehicle[:, 0] - box.center_x
    dy = points_vehicle[:, 1] - box.center_y
    dz = points_vehicle[:, 2] - box.center_z

    c = np.cos(box.heading)
    s = np.sin(box.heading)

    x_local = c * dx + s * dy
    y_local = -s * dx + c * dy

    return (
        (np.abs(x_local) <= box.length / 2.0 + margin)
        & (np.abs(y_local) <= box.width / 2.0 + margin)
        & (np.abs(dz) <= box.height / 2.0 + margin)
    )


def transform_points(points_xyz, T):
    if len(points_xyz) == 0:
        return np.empty((0, 3), dtype=np.float32)

    xyz_h = np.concatenate(
        [
            points_xyz.astype(np.float64),
            np.ones((len(points_xyz), 1), dtype=np.float64),
        ],
        axis=1,
    )

    return (xyz_h @ T.T)[:, :3].astype(np.float32)


def segmentation_proto_has_data(seg_proto):
    return (
        seg_proto is not None
        and hasattr(seg_proto, "data")
        and len(seg_proto.data) > 0
    )


def extract_labeled_top_frame(frame):
    """
    Extract labeled TOP points for every return for which segmentation exists.

    The same valid range-image mask is used for:
      XYZ
      intensity
      instance_id
      semantic_id

    Therefore the arrays remain pointwise aligned.
    """
    (
        range_images,
        camera_projections,
        segmentation_labels,
        range_image_top_pose,
    ) = frame_utils.parse_range_image_and_camera_projection(frame)

    top_name = open_dataset.LaserName.TOP

    if top_name not in segmentation_labels:
        return None

    if top_name not in range_images:
        return None

    top_seg_returns = segmentation_labels[top_name]
    top_range_returns = range_images[top_name]

    xyz_parts = []
    intensity_parts = []
    instance_parts = []
    semantic_parts = []
    return_parts = []
    row_parts = []
    col_parts = []

    for ri_index in [0, 1]:

        if ri_index >= len(top_seg_returns):
            continue

        if ri_index >= len(top_range_returns):
            continue

        seg_proto = top_seg_returns[ri_index]

        if not segmentation_proto_has_data(seg_proto):
            continue

        ri_proto = top_range_returns[ri_index]

        ri = tf.reshape(
            tf.convert_to_tensor(ri_proto.data),
            ri_proto.shape.dims,
        ).numpy()

        seg = tf.reshape(
            tf.convert_to_tensor(seg_proto.data),
            seg_proto.shape.dims,
        ).numpy()

        valid = ri[..., 0] > 0

        points, _ = frame_utils.convert_range_image_to_point_cloud(
            frame,
            range_images,
            camera_projections,
            range_image_top_pose,
            ri_index=ri_index,
        )

        xyz = np.asarray(points[0], dtype=np.float32)
        intensity = ri[..., 1][valid].astype(np.float32)
        valid_seg = seg[valid].astype(np.int32)
        pixel_rc = np.argwhere(valid).astype(np.int16)

        if not (
            len(xyz)
            == len(intensity)
            == len(valid_seg)
            == len(pixel_rc)
        ):
            raise RuntimeError(
                f"Point/label mismatch in return {ri_index}: "
                f"xyz={len(xyz)}, intensity={len(intensity)}, "
                f"seg={len(valid_seg)}, pixels={len(pixel_rc)}"
            )

        # Verified Waymo segmentation range-image convention:
        # channel 0 = instance ID
        # channel 1 = semantic ID
        instance_id = valid_seg[:, 0]
        semantic_id = valid_seg[:, 1]

        xyz_parts.append(xyz)
        intensity_parts.append(intensity)
        instance_parts.append(instance_id)
        semantic_parts.append(semantic_id)
        return_parts.append(
            np.full(len(xyz), ri_index, dtype=np.uint8)
        )
        row_parts.append(pixel_rc[:, 0])
        col_parts.append(pixel_rc[:, 1])

    if not xyz_parts:
        return None

    return {
        "xyz_vehicle": np.concatenate(xyz_parts, axis=0),
        "intensity": np.concatenate(intensity_parts, axis=0),
        "instance_id": np.concatenate(instance_parts, axis=0),
        "semantic_id": np.concatenate(semantic_parts, axis=0),
        "return_id": np.concatenate(return_parts, axis=0),
        "range_row": np.concatenate(row_parts, axis=0),
        "range_col": np.concatenate(col_parts, axis=0),
    }


def identify_dynamic_instance_ids(
    points_vehicle,
    instance_id,
    frame,
    dynamic_track_ids,
    box_margin=0.0,
):
    """
    Map each dynamic 3D box to the segmentation instance ID observed inside it.

    IMPORTANT:
    We DO NOT remove all points geometrically inside a dynamic box.

    Instead, for each known dynamic Waymo track:
      1. Look at segmentation instance IDs of points INSIDE its 3D box.
      2. Ignore instance_id == -1 (no instance).
      3. Choose the dominant valid instance ID in that box.
      4. Treat that segmentation instance as dynamic.

    After dynamic instance IDs are identified, ALL points carrying one of those
    instance IDs are removed from the static labeled accumulation. This also
    catches object points that fall slightly outside the nominal 3D box.

    Returns:
      is_dynamic_instance_point: bool[N]
      dynamic_instance_ids: sorted list[int]
      matches: diagnostic list[dict]
    """
    dynamic_instance_ids = set()
    matches = []

    for label in frame.laser_labels:

        if label.id not in dynamic_track_ids:
            continue

        inside = box_mask(
            points_vehicle,
            label.box,
            margin=box_margin,
        )

        inside_instance_ids = instance_id[inside]

        # -1 means no instance assignment in the Waymo segmentation.
        valid_instance_ids = inside_instance_ids[
            inside_instance_ids >= 0
        ]

        if len(valid_instance_ids) == 0:
            matches.append(
                {
                    "track_id": label.id,
                    "matched_instance_id": None,
                    "points_inside_box": int(np.count_nonzero(inside)),
                    "valid_instance_points_inside_box": 0,
                    "matched_instance_points_inside_box": 0,
                    "dominance_fraction": 0.0,
                }
            )
            continue

        ids, counts = np.unique(
            valid_instance_ids,
            return_counts=True,
        )

        best_idx = int(np.argmax(counts))
        matched_instance_id = int(ids[best_idx])
        matched_count = int(counts[best_idx])

        dynamic_instance_ids.add(
            matched_instance_id
        )

        matches.append(
            {
                "track_id": label.id,
                "matched_instance_id": matched_instance_id,
                "points_inside_box": int(np.count_nonzero(inside)),
                "valid_instance_points_inside_box": int(
                    len(valid_instance_ids)
                ),
                "matched_instance_points_inside_box": matched_count,
                "dominance_fraction": float(
                    matched_count / max(len(valid_instance_ids), 1)
                ),
            }
        )

    dynamic_instance_ids = sorted(dynamic_instance_ids)

    if dynamic_instance_ids:
        is_dynamic_instance_point = np.isin(
            instance_id,
            np.asarray(dynamic_instance_ids, dtype=np.int32),
        )
    else:
        is_dynamic_instance_point = np.zeros(
            len(points_vehicle),
            dtype=bool,
        )

    return (
        is_dynamic_instance_point,
        dynamic_instance_ids,
        matches,
    )


def load_frames(tfrecord_path):
    dataset = tf.data.TFRecordDataset(
        tfrecord_path,
        compression_type="",
    )

    for frame_idx, data in enumerate(dataset):
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        yield frame_idx, frame


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Extract Waymo TOP labeled frames and build a sparse accumulated "
            "STATIC labeled world using instance-aware dynamic removal."
        )
    )

    parser.add_argument("--tfrecord", required=True)
    parser.add_argument("--lidargs-root", required=True)
    parser.add_argument("--case", default=None)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument(
        "--box-margin",
        type=float,
        default=0.0,
        help=(
            "Margin used ONLY while identifying which segmentation instance "
            "belongs to each dynamic 3D box. Default: 0.0."
        ),
    )

    parser.add_argument(
        "--skip-existing-frames",
        action="store_true",
    )

    args = parser.parse_args()

    if args.case is None:
        case = os.path.basename(args.tfrecord)

        if case.endswith(".tfrecord"):
            case = case[:-len(".tfrecord")]
    else:
        case = args.case

    output_dir = os.path.abspath(args.output_dir)

    labeled_frames_dir = os.path.join(
        output_dir,
        "01_labeled_frames",
    )

    accumulated_dir = os.path.join(
        output_dir,
        "02_accumulated_labeled",
    )

    os.makedirs(
        labeled_frames_dir,
        exist_ok=True,
    )

    os.makedirs(
        accumulated_dir,
        exist_ok=True,
    )

    dynamic_track_ids, classification_path = load_dynamic_track_ids(
        args.lidargs_root,
        case,
    )

    print("Case:")
    print(case)
    print()
    print("Dynamic-track classification:")
    print(classification_path)
    print(
        "Whole tracks classified dynamic:",
        len(dynamic_track_ids),
    )
    print()

    labeled_frame_indices = []
    frame_manifest = {}

    accum_xyz_world = []
    accum_semantic = []
    accum_instance = []
    accum_frame_idx = []
    accum_return_id = []
    accum_source_index = []

    total_scene_frames = 0
    total_labeled_points = 0
    total_static_labeled_points = 0
    total_dynamic_instance_points = 0

    all_match_diagnostics = {}

    for frame_idx, frame in load_frames(
        args.tfrecord
    ):

        total_scene_frames += 1

        labeled = extract_labeled_top_frame(
            frame
        )

        if labeled is None:
            print(
                f"[{frame_idx:03d}] no TOP segmentation"
            )
            continue

        xyz_vehicle = labeled[
            "xyz_vehicle"
        ]

        instance_id = labeled[
            "instance_id"
        ]

        T_v2w = np.asarray(
            frame.pose.transform,
            dtype=np.float64,
        ).reshape(4, 4)

        xyz_world = transform_points(
            xyz_vehicle,
            T_v2w,
        )

        (
            is_dynamic,
            dynamic_instance_ids,
            matches,
        ) = identify_dynamic_instance_ids(
            xyz_vehicle,
            instance_id,
            frame,
            dynamic_track_ids,
            box_margin=args.box_margin,
        )

        is_static = ~is_dynamic

        frame_path = os.path.join(
            labeled_frames_dir,
            f"frame_{frame_idx:03d}.npz",
        )

        if not (
            args.skip_existing_frames
            and os.path.exists(frame_path)
        ):
            np.savez(
                frame_path,
                xyz_vehicle=xyz_vehicle.astype(
                    np.float32
                ),
                xyz_world=xyz_world.astype(
                    np.float32
                ),
                intensity=labeled[
                    "intensity"
                ].astype(np.float32),
                instance_id=instance_id.astype(
                    np.int32
                ),
                semantic_id=labeled[
                    "semantic_id"
                ].astype(np.int32),
                return_id=labeled[
                    "return_id"
                ].astype(np.uint8),
                range_row=labeled[
                    "range_row"
                ].astype(np.int16),
                range_col=labeled[
                    "range_col"
                ].astype(np.int16),
                is_dynamic_track_point=is_dynamic,
                dynamic_instance_ids=np.asarray(
                    dynamic_instance_ids,
                    dtype=np.int32,
                ),
                frame_idx=np.int32(frame_idx),
                timestamp_micros=np.int64(
                    frame.timestamp_micros
                ),
                vehicle_to_world=T_v2w.astype(
                    np.float64
                ),
            )

        n_all = len(xyz_vehicle)
        n_dynamic = int(
            np.count_nonzero(is_dynamic)
        )
        n_static = int(
            np.count_nonzero(is_static)
        )

        labeled_frame_indices.append(
            frame_idx
        )

        all_match_diagnostics[
            str(frame_idx)
        ] = matches

        frame_manifest[
            str(frame_idx)
        ] = {
            "timestamp_micros": int(
                frame.timestamp_micros
            ),
            "num_labeled_top_points": int(
                n_all
            ),
            "num_static_labeled_points": int(
                n_static
            ),
            "num_dynamic_instance_points": int(
                n_dynamic
            ),
            "dynamic_instance_ids": [
                int(x)
                for x in dynamic_instance_ids
            ],
            "file": os.path.relpath(
                frame_path,
                output_dir,
            ),
        }

        total_labeled_points += n_all
        total_static_labeled_points += n_static
        total_dynamic_instance_points += n_dynamic

        accum_xyz_world.append(
            xyz_world[is_static].astype(
                np.float32
            )
        )

        accum_semantic.append(
            labeled[
                "semantic_id"
            ][is_static].astype(
                np.int32
            )
        )

        accum_instance.append(
            instance_id[
                is_static
            ].astype(
                np.int32
            )
        )

        accum_frame_idx.append(
            np.full(
                n_static,
                frame_idx,
                dtype=np.int16,
            )
        )

        accum_return_id.append(
            labeled[
                "return_id"
            ][is_static].astype(
                np.uint8
            )
        )

        accum_source_index.append(
            np.flatnonzero(
                is_static
            ).astype(
                np.int32
            )
        )

        print(
            f"[{frame_idx:03d}] TOP segmentation | "
            f"all={n_all:,} | "
            f"static={n_static:,} | "
            f"dynamic-instance={n_dynamic:,} | "
            f"dynamic instance IDs={dynamic_instance_ids}"
        )

    if not labeled_frame_indices:
        raise RuntimeError(
            "No TOP segmentation frames found."
        )

    print()
    print(
        "Concatenating sparse labeled STATIC world reference..."
    )

    xyz_world = np.concatenate(
        accum_xyz_world,
        axis=0,
    )

    semantic_id = np.concatenate(
        accum_semantic,
        axis=0,
    )

    instance_id = np.concatenate(
        accum_instance,
        axis=0,
    )

    source_frame_idx = np.concatenate(
        accum_frame_idx,
        axis=0,
    )

    return_id = np.concatenate(
        accum_return_id,
        axis=0,
    )

    source_point_index = np.concatenate(
        accum_source_index,
        axis=0,
    )

    accumulated_path = os.path.join(
        accumulated_dir,
        "labeled_static_accumulated.npz",
    )

    np.savez(
        accumulated_path,
        xyz_world=xyz_world,
        semantic_id=semantic_id,
        instance_id=instance_id,
        source_frame_idx=source_frame_idx,
        return_id=return_id,
        source_point_index=source_point_index,
    )

    manifest = {
        "case": case,
        "num_scene_frames": int(
            total_scene_frames
        ),
        "num_labeled_frames": int(
            len(labeled_frame_indices)
        ),
        "labeled_frames": [
            int(x)
            for x in labeled_frame_indices
        ],
        "first_labeled_frame": int(
            min(labeled_frame_indices)
        ),
        "last_labeled_frame": int(
            max(labeled_frame_indices)
        ),
        "num_all_labeled_top_points": int(
            total_labeled_points
        ),
        "num_static_labeled_points": int(
            total_static_labeled_points
        ),
        "num_dynamic_instance_points_excluded": int(
            total_dynamic_instance_points
        ),
        "dynamic_track_count": int(
            len(dynamic_track_ids)
        ),
        "dynamic_removal_method": (
            "For each known dynamic 3D track box, find the dominant "
            "non-negative Waymo segmentation instance ID among points inside "
            "the box; then remove all points in that frame carrying one of "
            "those matched dynamic instance IDs."
        ),
        "frames": frame_manifest,
        "accumulated_static_file": os.path.relpath(
            accumulated_path,
            output_dir,
        ),
    }

    manifest_path = os.path.join(
        output_dir,
        "labeled_frame_indices.json",
    )

    with open(
        manifest_path,
        "w",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
        )

    diagnostics_path = os.path.join(
        output_dir,
        "dynamic_instance_matches.json",
    )

    with open(
        diagnostics_path,
        "w",
    ) as f:
        json.dump(
            all_match_diagnostics,
            f,
            indent=2,
        )

    print()
    print("=" * 72)
    print(
        "INSTANCE-AWARE LABELED REFERENCE COMPLETE"
    )
    print("=" * 72)

    print(
        "Scene frames                 :",
        total_scene_frames,
    )

    print(
        "Detected labeled frames      :",
        len(labeled_frame_indices),
    )

    print(
        "All labeled TOP points       :",
        f"{total_labeled_points:,}",
    )

    print(
        "Static labeled points        :",
        f"{total_static_labeled_points:,}",
    )

    print(
        "Dynamic instance points removed:",
        f"{total_dynamic_instance_points:,}",
    )

    print()
    print(
        "Sparse accumulated labeled static world:"
    )
    print(
        accumulated_path
    )

    print()
    print(
        "Dynamic instance matching diagnostics:"
    )
    print(
        diagnostics_path
    )


if __name__ == "__main__":
    main()