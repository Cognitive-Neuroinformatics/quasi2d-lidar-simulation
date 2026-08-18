import argparse
import os

import numpy as np
import tensorflow as tf
from scipy.spatial import cKDTree

from waymo_open_dataset import dataset_pb2
from waymo_open_dataset import label_pb2

from waymo_open_dataset.utils import frame_utils


# -------------------------------------------------------------------------
# Waymo semantic classes
# -------------------------------------------------------------------------

CLASS_NAMES = {
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


# Classes whose geometry may move independently of ego vehicle.
DYNAMIC_SEMANTIC_CLASSES = {
    1,   # CAR
    2,   # TRUCK
    3,   # BUS
    4,   # OTHER_VEHICLE
    5,   # MOTORCYCLIST
    6,   # BICYCLIST
    7,   # PEDESTRIAN
    12,  # BICYCLE
    13,  # MOTORCYCLE
}


# Waymo detection-label types that represent potentially moving objects.
DYNAMIC_BOX_TYPES = {
    label_pb2.Label.TYPE_VEHICLE,
    label_pb2.Label.TYPE_PEDESTRIAN,
    label_pb2.Label.TYPE_CYCLIST,
}

# -------------------------------------------------------------------------
# Basic helpers
# -------------------------------------------------------------------------

def get_pose(frame):
    return np.asarray(
        frame.pose.transform,
        dtype=np.float64,
    ).reshape(4, 4)


def transform_points(xyz, T):
    xyz_h = np.concatenate(
        [
            xyz,
            np.ones((len(xyz), 1), dtype=np.float64),
        ],
        axis=1,
    )

    return (T @ xyz_h.T).T[:, :3]


def ego_transform_source_to_target(
    xyz_source,
    source_pose,
    target_pose,
):
    """
    Source vehicle coordinates -> target vehicle coordinates.
    """

    T_source_to_target = (
        np.linalg.inv(target_pose)
        @ source_pose
    )

    return transform_points(
        xyz_source,
        T_source_to_target,
    )


# -------------------------------------------------------------------------
# Semantic-label availability
# -------------------------------------------------------------------------

def frame_has_semantics(frame):

    (
        _,
        _,
        segmentation_labels,
        _,
    ) = frame_utils.parse_range_image_and_camera_projection(frame)

    top = dataset_pb2.LaserName.TOP

    return (
        top in segmentation_labels
        and len(segmentation_labels[top]) > 0
    )


# -------------------------------------------------------------------------
# TOP LiDAR extraction
# -------------------------------------------------------------------------

def extract_point_labels(
    frame,
    range_images,
    segmentation_labels,
    ri_index,
):
    calibrations = sorted(
        frame.context.laser_calibrations,
        key=lambda c: c.name,
    )

    point_labels = []

    for calibration in calibrations:

        ri = range_images[
            calibration.name
        ][ri_index]

        ri_tensor = tf.reshape(
            tf.convert_to_tensor(ri.data),
            ri.shape.dims,
        )

        valid_mask = ri_tensor[..., 0] > 0

        if calibration.name in segmentation_labels:

            labels = segmentation_labels[
                calibration.name
            ][ri_index]

            labels_tensor = tf.reshape(
                tf.convert_to_tensor(labels.data),
                labels.shape.dims,
            )

            valid_labels = tf.gather_nd(
                labels_tensor,
                tf.where(valid_mask),
            )

        else:
            n = tf.reduce_sum(
                tf.cast(valid_mask, tf.int32)
            )

            valid_labels = tf.zeros(
                [n, 2],
                dtype=tf.int32,
            )

        point_labels.append(valid_labels.numpy())

    return point_labels


def extract_top_xyz(frame):
    """
    Extract TOP LiDAR XYZ for both returns.
    Works for labeled AND unlabeled frames.
    """

    (
        range_images,
        camera_projections,
        _,
        range_image_top_pose,
    ) = frame_utils.parse_range_image_and_camera_projection(frame)

    calibrations = sorted(
        frame.context.laser_calibrations,
        key=lambda c: c.name,
    )

    top = dataset_pb2.LaserName.TOP

    top_index = next(
        i
        for i, c in enumerate(calibrations)
        if c.name == top
    )

    xyz_all = []

    for ri_index in [0, 1]:

        points, _ = (
            frame_utils.convert_range_image_to_point_cloud(
                frame,
                range_images,
                camera_projections,
                range_image_top_pose,
                ri_index=ri_index,
            )
        )

        xyz_all.append(
            points[top_index].astype(np.float64)
        )

    return np.concatenate(
        xyz_all,
        axis=0,
    )


def extract_top_xyz_semantics(frame):
    """
    Extract XYZ + semantic + instance ID.
    Only for semantic-labelled frames.
    """

    (
        range_images,
        camera_projections,
        segmentation_labels,
        range_image_top_pose,
    ) = frame_utils.parse_range_image_and_camera_projection(frame)

    top = dataset_pb2.LaserName.TOP

    if (
        top not in segmentation_labels
        or len(segmentation_labels[top]) == 0
    ):
        raise RuntimeError(
            "Frame does not contain TOP semantic labels."
        )

    calibrations = sorted(
        frame.context.laser_calibrations,
        key=lambda c: c.name,
    )

    top_index = next(
        i
        for i, c in enumerate(calibrations)
        if c.name == top
    )

    xyz_all = []
    sem_all = []
    inst_all = []

    for ri_index in [0, 1]:

        points, _ = (
            frame_utils.convert_range_image_to_point_cloud(
                frame,
                range_images,
                camera_projections,
                range_image_top_pose,
                ri_index=ri_index,
            )
        )

        labels = extract_point_labels(
            frame,
            range_images,
            segmentation_labels,
            ri_index,
        )

        xyz = points[top_index]
        lab = labels[top_index]

        if len(xyz) != len(lab):
            raise RuntimeError(
                f"XYZ/label mismatch: "
                f"{len(xyz)} vs {len(lab)}"
            )

        xyz_all.append(xyz.astype(np.float64))
        inst_all.append(lab[:, 0].astype(np.int32))
        sem_all.append(lab[:, 1].astype(np.int32))

    return (
        np.concatenate(xyz_all),
        np.concatenate(sem_all),
        np.concatenate(inst_all),
    )


# -------------------------------------------------------------------------
# Waymo tracked boxes
# -------------------------------------------------------------------------

def get_dynamic_boxes(frame):
    """
    Return dictionary:
        track_id -> Waymo Label
    """

    result = {}

    for label in frame.laser_labels:

        if label.type not in DYNAMIC_BOX_TYPES:
            continue

        if not label.id:
            continue

        result[label.id] = label

    return result


# -------------------------------------------------------------------------
# Point-in-oriented-box
# -------------------------------------------------------------------------

def points_in_box(xyz, box, margin=0.0):

    center = np.array(
        [
            box.center_x,
            box.center_y,
            box.center_z,
        ],
        dtype=np.float64,
    )

    heading = box.heading

    c = np.cos(heading)
    s = np.sin(heading)

    # world/vehicle -> box local rotation
    R_inv = np.array(
        [
            [ c,  s, 0],
            [-s,  c, 0],
            [ 0,  0, 1],
        ],
        dtype=np.float64,
    )

    local = (
        R_inv @ (xyz - center).T
    ).T

    inside = (
        (np.abs(local[:, 0])
         <= box.length / 2 + margin)
        &
        (np.abs(local[:, 1])
         <= box.width / 2 + margin)
        &
        (np.abs(local[:, 2])
         <= box.height / 2 + margin)
    )

    return inside


# -------------------------------------------------------------------------
# Object-local motion
# -------------------------------------------------------------------------

def transform_object_points(
    xyz,
    source_box,
    target_box,
):
    """
    Convert points:
        source vehicle
          -> source object coordinates
          -> target vehicle coordinates

    This captures translation + heading change of the tracked object.
    """

    cs = np.array(
        [
            source_box.center_x,
            source_box.center_y,
            source_box.center_z,
        ],
        dtype=np.float64,
    )

    ct = np.array(
        [
            target_box.center_x,
            target_box.center_y,
            target_box.center_z,
        ],
        dtype=np.float64,
    )

    hs = source_box.heading
    ht = target_box.heading

    cs_h = np.cos(hs)
    ss_h = np.sin(hs)

    ct_h = np.cos(ht)
    st_h = np.sin(ht)

    R_source_inv = np.array(
        [
            [ cs_h,  ss_h, 0],
            [-ss_h,  cs_h, 0],
            [ 0,     0,    1],
        ],
        dtype=np.float64,
    )

    R_target = np.array(
        [
            [ct_h, -st_h, 0],
            [st_h,  ct_h, 0],
            [0,     0,    1],
        ],
        dtype=np.float64,
    )

    local = (
        R_source_inv
        @ (xyz - cs).T
    ).T

    return (
        R_target @ local.T
    ).T + ct


# -------------------------------------------------------------------------
# Find nearest labelled frame
# -------------------------------------------------------------------------

def nearest_labeled_frame(frame_idx, labeled_frames):

    return min(
        labeled_frames,
        key=lambda x: abs(x - frame_idx),
    )


# -------------------------------------------------------------------------
# Static propagation
# -------------------------------------------------------------------------

def propagate_static(
    source_xyz,
    source_semantic,
    source_pose,
    target_xyz,
    target_pose,
    target_dynamic_mask,
    max_distance,
):

    source_static_mask = (
        (source_semantic != 0)
        & ~np.isin(
            source_semantic,
            list(DYNAMIC_SEMANTIC_CLASSES),
        )
    )

    source_static_xyz = (
        source_xyz[source_static_mask]
    )

    source_static_sem = (
        source_semantic[source_static_mask]
    )

    source_aligned = (
        ego_transform_source_to_target(
            source_static_xyz,
            source_pose,
            target_pose,
        )
    )

    # Static labels should only be assigned outside
    # tracked dynamic objects.
    target_indices = np.where(
        ~target_dynamic_mask
    )[0]

    predicted = np.full(
        len(target_xyz),
        -1,
        dtype=np.int32,
    )

    distances = np.full(
        len(target_xyz),
        np.inf,
        dtype=np.float32,
    )

    if (
        len(source_aligned) == 0
        or len(target_indices) == 0
    ):
        return predicted, distances

    tree = cKDTree(source_aligned)

    d, idx = tree.query(
        target_xyz[target_indices],
        k=1,
        workers=-1,
    )

    valid = d <= max_distance

    valid_target = target_indices[valid]

    predicted[valid_target] = (
        source_static_sem[idx[valid]]
    )

    distances[valid_target] = d[valid]

    return predicted, distances


# -------------------------------------------------------------------------
# Dynamic propagation
# -------------------------------------------------------------------------

def propagate_dynamic(
    source_xyz,
    source_semantic,
    source_boxes,
    target_xyz,
    target_boxes,
    max_distance,
    box_margin,
):

    predicted = np.full(
        len(target_xyz),
        -1,
        dtype=np.int32,
    )

    distances = np.full(
        len(target_xyz),
        np.inf,
        dtype=np.float32,
    )

    matched_tracks = 0
    unmatched_tracks = 0

    for track_id, target_label in target_boxes.items():

        target_box = target_label.box

        target_inside = points_in_box(
            target_xyz,
            target_box,
            margin=box_margin,
        )

        target_indices = np.where(
            target_inside
        )[0]

        if len(target_indices) == 0:
            continue

        # We need the same object in the semantic-labelled source frame.
        if track_id not in source_boxes:

            unmatched_tracks += 1
            continue

        source_box = (
            source_boxes[track_id].box
        )

        source_inside = points_in_box(
            source_xyz,
            source_box,
            margin=box_margin,
        )

        # Only propagate known dynamic semantics.
        dynamic_semantic = np.isin(
            source_semantic,
            list(DYNAMIC_SEMANTIC_CLASSES),
        )

        source_mask = (
            source_inside
            & dynamic_semantic
        )

        source_object_xyz = (
            source_xyz[source_mask]
        )

        source_object_sem = (
            source_semantic[source_mask]
        )

        if len(source_object_xyz) == 0:
            unmatched_tracks += 1
            continue

        # Move the object from its source box pose
        # directly to its target box pose.
        predicted_object_xyz = (
            transform_object_points(
                source_object_xyz,
                source_box,
                target_box,
            )
        )

        tree = cKDTree(
            predicted_object_xyz
        )

        d, idx = tree.query(
            target_xyz[target_indices],
            k=1,
            workers=-1,
        )

        valid = d <= max_distance

        valid_target = (
            target_indices[valid]
        )

        predicted[valid_target] = (
            source_object_sem[
                idx[valid]
            ]
        )

        distances[valid_target] = (
            d[valid]
        )

        matched_tracks += 1

    return (
        predicted,
        distances,
        matched_tracks,
        unmatched_tracks,
    )


# -------------------------------------------------------------------------
# Dynamic target mask
# -------------------------------------------------------------------------

def build_dynamic_target_mask(
    target_xyz,
    target_boxes,
    margin,
):

    mask = np.zeros(
        len(target_xyz),
        dtype=bool,
    )

    for label in target_boxes.values():

        mask |= points_in_box(
            target_xyz,
            label.box,
            margin=margin,
        )

    return mask


# -------------------------------------------------------------------------
# First pass: labelled-frame indices
# -------------------------------------------------------------------------

def find_labeled_frames(tfrecord):

    print("Finding semantic-labelled frames...")

    dataset = tf.data.TFRecordDataset(
        tfrecord,
        compression_type="",
    )

    labeled = []

    for idx, data in enumerate(dataset):

        frame = dataset_pb2.Frame()
        frame.ParseFromString(
            bytearray(data.numpy())
        )

        if frame_has_semantics(frame):

            labeled.append(idx)

            print(
                f"  frame {idx:03d}"
            )

    return labeled


# -------------------------------------------------------------------------
# Second pass: cache labelled source frames
# -------------------------------------------------------------------------

def load_labeled_sources(
    tfrecord,
    labeled_frames,
):

    wanted = set(labeled_frames)

    cache = {}

    dataset = tf.data.TFRecordDataset(
        tfrecord,
        compression_type="",
    )

    print("\nCaching semantic-labelled source frames...")

    for idx, data in enumerate(dataset):

        if idx not in wanted:
            continue

        frame = dataset_pb2.Frame()

        frame.ParseFromString(
            bytearray(data.numpy())
        )

        xyz, semantic, instance = (
            extract_top_xyz_semantics(frame)
        )

        cache[idx] = {
            "xyz": xyz,
            "semantic": semantic,
            "instance": instance,
            "pose": get_pose(frame),
            "boxes": get_dynamic_boxes(frame),
        }

        print(
            f"  cached frame {idx:03d}: "
            f"{len(xyz)} points, "
            f"{len(cache[idx]['boxes'])} dynamic boxes"
        )

    return cache


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tfrecord",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--max-distance",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--box-margin",
        type=float,
        default=0.10,
        help="Expansion around dynamic Waymo boxes.",
    )

    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    # -------------------------------------------------------------
    # Pass 1
    # -------------------------------------------------------------

    labeled_frames = find_labeled_frames(
        args.tfrecord
    )

    if not labeled_frames:

        raise RuntimeError(
            "No semantic-labelled frames found."
        )

    print(
        "\nLabeled frames:",
        labeled_frames,
    )

    # -------------------------------------------------------------
    # Pass 2
    # -------------------------------------------------------------

    source_cache = load_labeled_sources(
        args.tfrecord,
        labeled_frames,
    )

    # -------------------------------------------------------------
    # Pass 3: process every frame
    # -------------------------------------------------------------

    dataset = tf.data.TFRecordDataset(
        args.tfrecord,
        compression_type="",
    )

    print("\nPropagating whole scene...\n")

    labeled_set = set(
        labeled_frames
    )

    for frame_idx, data in enumerate(dataset):

        if frame_idx < args.start_frame:
            continue

        if (
            args.end_frame is not None
            and frame_idx > args.end_frame
        ):
            break

        frame = dataset_pb2.Frame()

        frame.ParseFromString(
            bytearray(data.numpy())
        )

        # ---------------------------------------------------------
        # Actual GT frame
        # ---------------------------------------------------------

        if frame_idx in labeled_set:

            source = source_cache[
                frame_idx
            ]

            xyz = source["xyz"].astype(
                np.float32
            )

            semantic = source[
                "semantic"
            ].astype(np.int32)

            instance = source[
                "instance"
            ].astype(np.int32)

            nn_distance = np.zeros(
                len(xyz),
                dtype=np.float32,
            )

            propagation_type = np.full(
                len(xyz),
                0,
                dtype=np.int8,
            )

            source_idx = frame_idx

            is_gt = True

            print(
                f"Frame {frame_idx:03d}: "
                f"GROUND TRUTH"
            )

        # ---------------------------------------------------------
        # Unlabelled frame
        # ---------------------------------------------------------

        else:

            xyz = extract_top_xyz(
                frame
            )

            source_idx = (
                nearest_labeled_frame(
                    frame_idx,
                    labeled_frames,
                )
            )

            source = source_cache[
                source_idx
            ]

            target_pose = get_pose(
                frame
            )

            target_boxes = (
                get_dynamic_boxes(frame)
            )

            target_dynamic_mask = (
                build_dynamic_target_mask(
                    xyz,
                    target_boxes,
                    args.box_margin,
                )
            )

            # -----------------------------------------------------
            # Static world
            # -----------------------------------------------------

            static_pred, static_dist = (
                propagate_static(
                    source["xyz"],
                    source["semantic"],
                    source["pose"],
                    xyz,
                    target_pose,
                    target_dynamic_mask,
                    args.max_distance,
                )
            )

            # -----------------------------------------------------
            # Dynamic objects
            # -----------------------------------------------------

            (
                dynamic_pred,
                dynamic_dist,
                matched_tracks,
                unmatched_tracks,
            ) = propagate_dynamic(
                source["xyz"],
                source["semantic"],
                source["boxes"],
                xyz,
                target_boxes,
                args.max_distance,
                args.box_margin,
            )

            # -----------------------------------------------------
            # Merge
            # -----------------------------------------------------

            semantic = static_pred.copy()

            nn_distance = (
                static_dist.copy()
            )

            # 0 = unknown/GT
            # 1 = static propagation
            # 2 = dynamic object propagation

            propagation_type = np.zeros(
                len(xyz),
                dtype=np.int8,
            )

            static_valid = (
                static_pred >= 0
            )

            propagation_type[
                static_valid
            ] = 1

            dynamic_valid = (
                dynamic_pred >= 0
            )

            semantic[
                dynamic_valid
            ] = dynamic_pred[
                dynamic_valid
            ]

            nn_distance[
                dynamic_valid
            ] = dynamic_dist[
                dynamic_valid
            ]

            propagation_type[
                dynamic_valid
            ] = 2

            # Convert unassigned -1 -> class 0
            semantic[
                semantic < 0
            ] = 0

            instance = np.zeros(
                len(xyz),
                dtype=np.int32,
            )

            is_gt = False

            assigned = (
                semantic != 0
            )

            coverage = (
                np.mean(assigned)
                * 100.0
            )

            dynamic_points = np.sum(
                propagation_type == 2
            )

            print(
                f"Frame {frame_idx:03d}: "
                f"source={source_idx:03d} | "
                f"coverage={coverage:6.2f}% | "
                f"dynamic_points={dynamic_points:6d} | "
                f"tracks={matched_tracks}/{matched_tracks + unmatched_tracks}"
            )

        # ---------------------------------------------------------
        # Save
        # ---------------------------------------------------------

        output_path = os.path.join(
            args.output_dir,
            f"frame_{frame_idx:03d}.npz",
        )

        np.savez_compressed(
            output_path,
            xyz=xyz.astype(np.float32),
            semantic_class=semantic.astype(
                np.int16
            ),
            instance_id=instance.astype(
                np.int32
            ),
            nn_distance=nn_distance.astype(
                np.float32
            ),
            propagation_type=propagation_type,
            source_frame=np.int32(
                source_idx
            ),
            is_ground_truth=np.bool_(
                is_gt
            ),
        )


if __name__ == "__main__":
    main()