import argparse

import numpy as np
import tensorflow as tf

from scipy.spatial import cKDTree

from waymo_open_dataset import dataset_pb2
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


# -------------------------------------------------------------------------
# Load selected frames
# -------------------------------------------------------------------------

def load_frames(tfrecord_path, requested_indices):
    """
    Read only the requested frames from the TFRecord.
    """

    requested_indices = set(requested_indices)
    frames = {}

    dataset = tf.data.TFRecordDataset(
        tfrecord_path,
        compression_type=""
    )

    for frame_idx, data in enumerate(dataset):

        if frame_idx not in requested_indices:
            continue

        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        frames[frame_idx] = frame

        print(f"Loaded frame {frame_idx}")

        if len(frames) == len(requested_indices):
            break

    missing = requested_indices - set(frames.keys())

    if missing:
        raise RuntimeError(
            f"Could not find frames: {sorted(missing)}"
        )

    return frames


# -------------------------------------------------------------------------
# Semantic-label extraction
# -------------------------------------------------------------------------

def convert_range_image_to_point_cloud_labels(
    frame,
    range_images,
    segmentation_labels,
    ri_index=0,
):
    """
    Convert range-image semantic labels to point labels.

    This follows Waymo's official semantic segmentation tutorial.

    Returns:
        point_labels:
            list ordered according to sorted LiDAR calibrations.

            Each element is shape (N, 2):
                [:, 0] = instance ID
                [:, 1] = semantic class
    """

    calibrations = sorted(
        frame.context.laser_calibrations,
        key=lambda c: c.name,
    )

    point_labels = []

    for calibration in calibrations:

        range_image = range_images[calibration.name][ri_index]

        range_image_tensor = tf.reshape(
            tf.convert_to_tensor(range_image.data),
            range_image.shape.dims,
        )

        # EXACTLY the same validity criterion used for the XYZ points.
        range_image_mask = range_image_tensor[..., 0] > 0

        if calibration.name in segmentation_labels:

            semantic_label = (
                segmentation_labels[calibration.name][ri_index]
            )

            semantic_label_tensor = tf.reshape(
                tf.convert_to_tensor(semantic_label.data),
                semantic_label.shape.dims,
            )

            semantic_points = tf.gather_nd(
                semantic_label_tensor,
                tf.where(range_image_mask),
            )

        else:

            num_valid_points = tf.reduce_sum(
                tf.cast(range_image_mask, tf.int32)
            )

            semantic_points = tf.zeros(
                [num_valid_points, 2],
                dtype=tf.int32,
            )

        point_labels.append(semantic_points.numpy())

    return point_labels


# -------------------------------------------------------------------------
# Extract TOP LiDAR + semantic labels
# -------------------------------------------------------------------------

def extract_top_lidar(frame):
    """
    Extract TOP-LiDAR XYZ and semantic labels for both returns.

    Returns:
        xyz              (N, 3)
        semantic_class   (N,)
        instance_id      (N,)
    """

    (
        range_images,
        camera_projections,
        segmentation_labels,
        range_image_top_pose,
    ) = frame_utils.parse_range_image_and_camera_projection(frame)

    top_name = dataset_pb2.LaserName.TOP

    if (
        top_name not in segmentation_labels
        or len(segmentation_labels[top_name]) == 0
    ):
        raise RuntimeError(
            "This frame does not contain TOP-LiDAR semantic labels."
        )

    all_xyz = []
    all_semantic = []
    all_instance = []

    # Waymo has two LiDAR returns.
    for ri_index in [0, 1]:

        # -------------------------------------------------------------
        # XYZ points
        # -------------------------------------------------------------

        points, _ = frame_utils.convert_range_image_to_point_cloud(
            frame,
            range_images,
            camera_projections,
            range_image_top_pose,
            ri_index=ri_index,
        )

        # -------------------------------------------------------------
        # Semantic labels
        # -------------------------------------------------------------

        point_labels = convert_range_image_to_point_cloud_labels(
            frame,
            range_images,
            segmentation_labels,
            ri_index=ri_index,
        )

        # frame_utils returns lists ordered by sorted calibration names.
        calibrations = sorted(
            frame.context.laser_calibrations,
            key=lambda c: c.name,
        )

        top_index = None

        for i, calibration in enumerate(calibrations):
            if calibration.name == top_name:
                top_index = i
                break

        if top_index is None:
            raise RuntimeError("TOP LiDAR calibration not found.")

        xyz = points[top_index]
        labels = point_labels[top_index]

        if xyz.shape[0] != labels.shape[0]:
            raise RuntimeError(
                "XYZ/label mismatch: "
                f"{xyz.shape[0]} points vs "
                f"{labels.shape[0]} labels"
            )

        all_xyz.append(xyz.astype(np.float64))
        all_instance.append(labels[:, 0].astype(np.int32))
        all_semantic.append(labels[:, 1].astype(np.int32))

        print(
            f"  Return {ri_index}: "
            f"{xyz.shape[0]} valid TOP points"
        )

    xyz = np.concatenate(all_xyz, axis=0)
    instance_id = np.concatenate(all_instance, axis=0)
    semantic_class = np.concatenate(all_semantic, axis=0)

    return xyz, semantic_class, instance_id


# -------------------------------------------------------------------------
# Pose utilities
# -------------------------------------------------------------------------

def get_vehicle_to_world(frame):
    """
    Waymo frame.pose is a row-major 4x4 transform:
        vehicle frame -> world/global frame.
    """

    return np.array(
        frame.pose.transform,
        dtype=np.float64,
    ).reshape(4, 4)


def transform_points(points, transform):
    """
    Apply homogeneous 4x4 transformation to Nx3 points.
    """

    ones = np.ones(
        (points.shape[0], 1),
        dtype=np.float64,
    )

    homogeneous = np.concatenate(
        [points, ones],
        axis=1,
    )

    transformed = (
        transform @ homogeneous.T
    ).T

    return transformed[:, :3]


def transform_source_to_target(
    xyz_source,
    frame_source,
    frame_target,
):
    """
    Transform source-frame LiDAR points into the target vehicle frame.

    p_target =
        inv(T_target_world) @
        T_source_world @
        p_source
    """

    T_source_world = get_vehicle_to_world(frame_source)
    T_target_world = get_vehicle_to_world(frame_target)

    T_source_to_target = (
        np.linalg.inv(T_target_world)
        @ T_source_world
    )

    xyz_source_in_target = transform_points(
        xyz_source,
        T_source_to_target,
    )

    return xyz_source_in_target, T_source_to_target


# -------------------------------------------------------------------------
# Nearest-neighbor propagation
# -------------------------------------------------------------------------

def propagate_nearest_neighbor(
    source_xyz,
    source_labels,
    target_xyz,
    max_distance,
):
    """
    Assign target point the semantic class of its nearest
    source point.

    Points whose nearest source point is farther than max_distance
    remain unassigned (-1).
    """

    print("\nBuilding KD-tree...")

    tree = cKDTree(source_xyz)

    print("Querying nearest neighbors...")

    distances, indices = tree.query(
        target_xyz,
        k=1,
        workers=-1,
    )

    predicted = np.full(
        target_xyz.shape[0],
        -1,
        dtype=np.int32,
    )

    valid = distances <= max_distance

    predicted[valid] = source_labels[indices[valid]]

    return predicted, distances, indices, valid


# -------------------------------------------------------------------------
# Metrics
# -------------------------------------------------------------------------

def evaluate_predictions(
    predicted,
    ground_truth,
    assigned_mask,
):
    """
    Evaluate only points that received a propagated label.

    GT class 0 (UNDEFINED / NOT LABELED) is excluded.
    """

    gt_valid = ground_truth != 0

    evaluation_mask = assigned_mask & gt_valid

    total_target = np.sum(gt_valid)
    total_evaluated = np.sum(evaluation_mask)

    coverage = (
        total_evaluated / total_target
        if total_target > 0
        else 0
    )

    correct = (
        predicted[evaluation_mask]
        == ground_truth[evaluation_mask]
    )

    accuracy = (
        np.mean(correct)
        if total_evaluated > 0
        else 0
    )

    print("\n========================================")
    print("Evaluation")
    print("========================================")

    print(
        f"GT labeled target points : {total_target}"
    )

    print(
        f"Propagated points        : {total_evaluated}"
    )

    print(
        f"Coverage                 : "
        f"{coverage * 100:.2f}%"
    )

    print(
        f"Semantic accuracy        : "
        f"{accuracy * 100:.2f}%"
    )

    # -------------------------------------------------------------
    # Per-class IoU
    # -------------------------------------------------------------

    ious = []

    print("\nPer-class IoU")
    print("----------------------------------------")

    for class_id in range(1, 23):

        gt_class = (
            ground_truth[evaluation_mask] == class_id
        )

        pred_class = (
            predicted[evaluation_mask] == class_id
        )

        intersection = np.sum(
            gt_class & pred_class
        )

        union = np.sum(
            gt_class | pred_class
        )

        if union == 0:
            continue

        iou = intersection / union
        ious.append(iou)

        class_name = CLASS_NAMES.get(
            class_id,
            f"CLASS_{class_id}"
        )

        print(
            f"{class_id:2d} "
            f"{class_name:20s} "
            f"IoU = {iou:.4f}"
        )

    miou = np.mean(ious) if ious else 0.0

    print("----------------------------------------")
    print(f"mIoU = {miou:.4f}")

    return {
        "coverage": coverage,
        "accuracy": accuracy,
        "miou": miou,
    }


# -------------------------------------------------------------------------
# Distance statistics
# -------------------------------------------------------------------------

def print_distance_statistics(
    distances,
    predicted,
    ground_truth,
    assigned_mask,
):
    """
    Show semantic accuracy as a function of NN distance.
    """

    bins = [
        (0.00, 0.05),
        (0.05, 0.10),
        (0.10, 0.20),
        (0.20, 0.30),
        (0.30, 0.50),
        (0.50, 1.00),
    ]

    print("\n========================================")
    print("Accuracy vs nearest-neighbor distance")
    print("========================================")

    for d_min, d_max in bins:

        mask = (
            assigned_mask
            & (ground_truth != 0)
            & (distances >= d_min)
            & (distances < d_max)
        )

        count = np.sum(mask)

        if count == 0:
            continue

        accuracy = np.mean(
            predicted[mask]
            == ground_truth[mask]
        )

        print(
            f"{d_min:4.2f} - {d_max:4.2f} m : "
            f"{count:7d} points, "
            f"accuracy = {accuracy * 100:6.2f}%"
        )


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tfrecord",
        required=True,
        help="Path to Waymo TFRecord.",
    )

    parser.add_argument(
        "--source-frame",
        type=int,
        default=25,
        help="Labeled source frame.",
    )

    parser.add_argument(
        "--target-frame",
        type=int,
        default=30,
        help="Labeled target frame used as hidden GT.",
    )

    parser.add_argument(
        "--max-distance",
        type=float,
        default=0.20,
        help=(
            "Maximum NN distance in meters. "
            "Target points farther away remain unassigned."
        ),
    )

    parser.add_argument(
        "--no-ego-compensation",
        action="store_true",
        help=(
            "Do not transform source points into target frame. "
            "Useful as Baseline 0."
        ),
    )

    args = parser.parse_args()

    print("\nLoading frames...")

    frames = load_frames(
        args.tfrecord,
        [
            args.source_frame,
            args.target_frame,
        ],
    )

    source_frame = frames[args.source_frame]
    target_frame = frames[args.target_frame]

    # -------------------------------------------------------------
    # Extract source
    # -------------------------------------------------------------

    print(
        f"\nExtracting frame {args.source_frame}..."
    )

    (
        source_xyz,
        source_semantic,
        source_instance,
    ) = extract_top_lidar(source_frame)

    print(
        f"Total source points: {source_xyz.shape[0]}"
    )

    # -------------------------------------------------------------
    # Extract target
    # -------------------------------------------------------------

    print(
        f"\nExtracting frame {args.target_frame}..."
    )

    (
        target_xyz,
        target_semantic,
        target_instance,
    ) = extract_top_lidar(target_frame)

    print(
        f"Total target points: {target_xyz.shape[0]}"
    )

    # -------------------------------------------------------------
    # Ego-motion compensation
    # -------------------------------------------------------------

    if args.no_ego_compensation:

        print(
            "\nNO ego-motion compensation."
        )

        source_xyz_aligned = source_xyz

    else:

        print(
            "\nTransforming source cloud "
            "into target vehicle frame..."
        )

        (
            source_xyz_aligned,
            T_source_to_target,
        ) = transform_source_to_target(
            source_xyz,
            source_frame,
            target_frame,
        )

        print("\nT_source_to_target:")
        print(T_source_to_target)

    # -------------------------------------------------------------
    # NN propagation
    # -------------------------------------------------------------

    (
        predicted,
        distances,
        indices,
        assigned_mask,
    ) = propagate_nearest_neighbor(
        source_xyz_aligned,
        source_semantic,
        target_xyz,
        args.max_distance,
    )

    # -------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------

    evaluate_predictions(
        predicted,
        target_semantic,
        assigned_mask,
    )

    print_distance_statistics(
        distances,
        predicted,
        target_semantic,
        assigned_mask,
    )


if __name__ == "__main__":
    main()