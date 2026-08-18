import argparse
import numpy as np
import open3d as o3d
import tensorflow as tf

from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.utils import frame_utils


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


CLASS_COLORS = {
    0:  [0.4, 0.4, 0.4],
    1:  [1.0, 0.0, 0.0],
    2:  [0.8, 0.2, 0.0],
    3:  [0.7, 0.0, 0.2],
    4:  [0.6, 0.2, 0.2],
    5:  [1.0, 0.4, 0.0],
    6:  [1.0, 0.7, 0.0],
    7:  [1.0, 0.0, 1.0],
    8:  [0.0, 0.0, 1.0],
    9:  [1.0, 1.0, 0.0],
    10: [0.4, 0.4, 1.0],
    11: [1.0, 0.5, 0.0],
    12: [0.9, 0.8, 0.1],
    13: [0.8, 0.5, 0.0],
    14: [0.6, 0.6, 0.6],
    15: [0.0, 0.7, 0.0],
    16: [0.4, 0.25, 0.1],
    17: [0.0, 1.0, 1.0],
    18: [0.2, 0.2, 0.2],
    19: [1.0, 1.0, 1.0],
    20: [0.5, 0.4, 0.3],
    21: [0.3, 0.8, 0.8],
    22: [0.7, 0.7, 0.3],
}


def has_semantic_labels(frame):
    (
        range_images,
        camera_projections,
        segmentation_labels,
        range_image_top_pose,
    ) = frame_utils.parse_range_image_and_camera_projection(frame)

    top_name = dataset_pb2.LaserName.TOP

    return (
        top_name in segmentation_labels
        and len(segmentation_labels[top_name]) > 0
    )


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

        range_image = range_images[
            calibration.name
        ][ri_index]

        range_tensor = tf.reshape(
            tf.convert_to_tensor(range_image.data),
            range_image.shape.dims,
        )

        valid_mask = range_tensor[..., 0] > 0

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

            num_valid = tf.reduce_sum(
                tf.cast(valid_mask, tf.int32)
            )

            valid_labels = tf.zeros(
                [num_valid, 2],
                dtype=tf.int32,
            )

        point_labels.append(
            valid_labels.numpy()
        )

    return point_labels


def extract_top_lidar_semantics(frame):

    (
        range_images,
        camera_projections,
        segmentation_labels,
        range_image_top_pose,
    ) = frame_utils.parse_range_image_and_camera_projection(
        frame
    )

    top_name = dataset_pb2.LaserName.TOP

    if (
        top_name not in segmentation_labels
        or len(segmentation_labels[top_name]) == 0
    ):
        return None

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
        raise RuntimeError(
            "TOP LiDAR calibration not found."
        )

    all_xyz = []
    all_semantic = []
    all_instance = []

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

        point_labels = extract_point_labels(
            frame,
            range_images,
            segmentation_labels,
            ri_index,
        )

        xyz = points[top_index]
        labels = point_labels[top_index]

        if xyz.shape[0] != labels.shape[0]:
            raise RuntimeError(
                f"XYZ/label mismatch for return {ri_index}: "
                f"{xyz.shape[0]} vs {labels.shape[0]}"
            )

        all_xyz.append(
            xyz.astype(np.float64)
        )

        all_instance.append(
            labels[:, 0].astype(np.int32)
        )

        all_semantic.append(
            labels[:, 1].astype(np.int32)
        )

    xyz = np.concatenate(
        all_xyz,
        axis=0,
    )

    semantic = np.concatenate(
        all_semantic,
        axis=0,
    )

    instance = np.concatenate(
        all_instance,
        axis=0,
    )

    return xyz, semantic, instance


def semantic_colors(labels):

    colors = np.zeros(
        (labels.shape[0], 3),
        dtype=np.float64,
    )

    for class_id in np.unique(labels):

        mask = labels == class_id

        colors[mask] = CLASS_COLORS.get(
            int(class_id),
            [1.0, 1.0, 1.0],
        )

    return colors


def show_frame(
    xyz,
    semantic,
    frame_idx,
    point_size,
    hide_undefined,
):

    if hide_undefined:

        mask = semantic != 0

        xyz = xyz[mask]
        semantic = semantic[mask]

    print("\n========================================")
    print(f"Frame {frame_idx}")
    print("========================================")

    classes, counts = np.unique(
        semantic,
        return_counts=True,
    )

    for cls, count in zip(classes, counts):
        print(
            f"{int(cls):2d} "
            f"{CLASS_NAMES.get(int(cls), 'UNKNOWN'):20s} "
            f"{count:8d}"
        )

    colors = semantic_colors(
        semantic
    )

    pcd = o3d.geometry.PointCloud()

    pcd.points = (
        o3d.utility.Vector3dVector(
            xyz
        )
    )

    pcd.colors = (
        o3d.utility.Vector3dVector(
            colors
        )
    )

    vis = o3d.visualization.Visualizer()

    vis.create_window(
        window_name=(
            f"Waymo Semantic Frame {frame_idx}"
        ),
        width=1600,
        height=900,
    )

    vis.add_geometry(pcd)

    render = vis.get_render_option()

    render.point_size = point_size

    render.background_color = np.array(
        [0.05, 0.05, 0.05]
    )

    vis.run()
    vis.destroy_window()


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tfrecord",
        required=True,
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

    parser.add_argument(
        "--point-size",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--hide-undefined",
        action="store_true",
    )

    args = parser.parse_args()

    dataset = tf.data.TFRecordDataset(
        args.tfrecord,
        compression_type=""
    )

    labeled_frames = []

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

        if not has_semantic_labels(frame):

            print(
                f"Frame {frame_idx:03d}: "
                f"no semantic labels -> skipping"
            )

            continue

        print(
            f"\nFrame {frame_idx:03d}: "
            f"semantic labels FOUND"
        )

        result = extract_top_lidar_semantics(
            frame
        )

        if result is None:
            continue

        xyz, semantic, instance = result

        print(
            f"TOP LiDAR points: {xyz.shape[0]}"
        )

        labeled_frames.append(
            frame_idx
        )

        show_frame(
            xyz,
            semantic,
            frame_idx,
            args.point_size,
            args.hide_undefined,
        )

    print("\n========================================")
    print("Finished")
    print("========================================")

    print(
        "Labeled frames:",
        labeled_frames
    )

    print(
        "Number of labeled frames:",
        len(labeled_frames)
    )


if __name__ == "__main__":
    main()