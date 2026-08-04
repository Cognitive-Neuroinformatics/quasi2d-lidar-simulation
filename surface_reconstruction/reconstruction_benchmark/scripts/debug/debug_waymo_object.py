#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
import yaml
from google.protobuf.json_format import MessageToDict

from waymo_open_dataset import dataset_pb2
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import segmentation_pb2
from waymo_open_dataset.utils import frame_utils


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML configuration: {path}")

    return config


def load_frame(
    tfrecord_path: Path,
    frame_index: int,
) -> dataset_pb2.Frame:
    dataset = tf.data.TFRecordDataset(
        str(tfrecord_path),
        compression_type="",
    )

    for index, record in enumerate(dataset):
        if index != frame_index:
            continue

        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(record.numpy()))
        return frame

    raise RuntimeError(
        f"Frame {frame_index} was not found in {tfrecord_path}"
    )


def label_type_name(value: int) -> str:
    descriptor = label_pb2.Label.Type.DESCRIPTOR
    enum_value = descriptor.values_by_number.get(value)

    if enum_value is None:
        return f"UNKNOWN_LABEL_TYPE_{value}"

    return enum_value.name


def semantic_type_name(value: int) -> str:
    descriptor = segmentation_pb2.Segmentation.Type.DESCRIPTOR
    enum_value = descriptor.values_by_number.get(value)

    if enum_value is None:
        return f"UNKNOWN_SEMANTIC_TYPE_{value}"

    return enum_value.name


def laser_name(value: int) -> str:
    mapping = {
        dataset_pb2.LaserName.TOP: "TOP",
        dataset_pb2.LaserName.FRONT: "FRONT",
        dataset_pb2.LaserName.SIDE_LEFT: "SIDE_LEFT",
        dataset_pb2.LaserName.SIDE_RIGHT: "SIDE_RIGHT",
        dataset_pb2.LaserName.REAR: "REAR",
    }

    return mapping.get(value, f"UNKNOWN_LIDAR_{value}")


def rotation_z(heading: float) -> np.ndarray:
    c = np.cos(heading)
    s = np.sin(heading)

    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def points_inside_label_box(
    xyz_vehicle: np.ndarray,
    label: label_pb2.Label,
    margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    box = label.box

    center = np.array(
        [
            box.center_x,
            box.center_y,
            box.center_z,
        ],
        dtype=np.float64,
    )

    rotation_object_to_vehicle = rotation_z(
        float(box.heading)
    )

    xyz_object = (
        xyz_vehicle - center[None, :]
    ) @ rotation_object_to_vehicle

    mask = (
        (
            np.abs(xyz_object[:, 0])
            <= 0.5 * float(box.length) + margin
        )
        & (
            np.abs(xyz_object[:, 1])
            <= 0.5 * float(box.width) + margin
        )
        & (
            np.abs(xyz_object[:, 2])
            <= 0.5 * float(box.height) + margin
        )
    )

    return mask, xyz_object


def convert_segmentation_labels_to_points(
    frame: dataset_pb2.Frame,
    range_images: dict,
    segmentation_labels: dict,
    return_index: int,
) -> list[np.ndarray]:
    """
    Convert range-image segmentation labels into the same point order used
    by frame_utils.convert_range_image_to_point_cloud().

    Each returned point label has:
        column 0: instance ID
        column 1: semantic class
    """

    calibrations = sorted(
        frame.context.laser_calibrations,
        key=lambda calibration: calibration.name,
    )

    labels_per_lidar: list[np.ndarray] = []

    for calibration in calibrations:
        range_image = range_images[
            calibration.name
        ][return_index]

        range_image_tensor = tf.reshape(
            tf.convert_to_tensor(
                range_image.data
            ),
            range_image.shape.dims,
        )

        valid_mask = (
            range_image_tensor[..., 0] > 0
        )

        if (
            calibration.name
            in segmentation_labels
            and len(
                segmentation_labels[
                    calibration.name
                ]
            ) > return_index
        ):
            segmentation_proto = (
                segmentation_labels[
                    calibration.name
                ][return_index]
            )

            segmentation_tensor = tf.reshape(
                tf.convert_to_tensor(
                    segmentation_proto.data
                ),
                segmentation_proto.shape.dims,
            )

            point_labels = tf.gather_nd(
                segmentation_tensor,
                tf.where(valid_mask),
            ).numpy()
        else:
            count = int(
                tf.reduce_sum(
                    tf.cast(
                        valid_mask,
                        tf.int32,
                    )
                ).numpy()
            )

            point_labels = np.zeros(
                (count, 2),
                dtype=np.int32,
            )

        labels_per_lidar.append(
            np.asarray(
                point_labels,
                dtype=np.int32,
            )
        )

    return labels_per_lidar


def print_label_details(
    label: label_pb2.Label,
) -> None:
    box = label.box

    print("\nDetection/tracking label")
    print("------------------------")
    print(f"Track ID: {label.id}")
    print(
        f"Type: {label_type_name(label.type)} "
        f"({label.type})"
    )

    print("\n3D box in vehicle coordinates")
    print("-----------------------------")
    print(
        "Center: "
        f"[{box.center_x:.4f}, "
        f"{box.center_y:.4f}, "
        f"{box.center_z:.4f}]"
    )
    print(
        "Dimensions L/W/H: "
        f"{box.length:.4f} / "
        f"{box.width:.4f} / "
        f"{box.height:.4f} m"
    )
    print(
        f"Heading: {box.heading:.6f} rad"
    )

    print("\nLabel statistics")
    print("----------------")
    print(
        "num_lidar_points_in_box: "
        f"{label.num_lidar_points_in_box}"
    )

    if hasattr(
        label,
        "num_top_lidar_points_in_box",
    ):
        print(
            "num_top_lidar_points_in_box: "
            f"{label.num_top_lidar_points_in_box}"
        )

    print(
        "Detection difficulty: "
        f"{label.detection_difficulty_level}"
    )
    print(
        "Tracking difficulty: "
        f"{label.tracking_difficulty_level}"
    )

    print("\nMotion metadata")
    print("---------------")

    metadata = label.metadata

    for name in (
        "speed_x",
        "speed_y",
        "speed_z",
        "accel_x",
        "accel_y",
        "accel_z",
    ):
        if hasattr(metadata, name):
            print(
                f"{name}: "
                f"{getattr(metadata, name):.6f}"
            )

    if hasattr(
        label,
        "most_visible_camera_name",
    ):
        print(
            "\nMost visible camera: "
            f"{label.most_visible_camera_name}"
        )

    print("\nComplete protobuf contents")
    print("--------------------------")

    label_dict = MessageToDict(
        label,
        preserving_proto_field_name=True,
    )

    print(
        json.dumps(
            label_dict,
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a Waymo laser object label, its 3D box crop, "
            "and any point-level segmentation labels."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--frame-index",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--track-id",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--lidar",
        type=str,
        default="TOP",
    )

    parser.add_argument(
        "--return-index",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--box-margin",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional NPZ output containing the full selected LiDAR "
            "cloud, box mask and point-level segmentation labels."
        ),
    )

    args = parser.parse_args()

    config = load_yaml(args.config)
    input_config = config["input"]

    tfrecord_path = Path(
        input_config["tfrecord"]
    ).expanduser().resolve()

    return_index = (
        int(
            input_config.get(
                "return_index",
                0,
            )
        )
        if args.return_index is None
        else args.return_index
    )

    if return_index not in (0, 1):
        raise ValueError(
            "return-index must be 0 or 1"
        )

    requested_lidar = (
        args.lidar.strip().upper()
    )

    lidar_mapping = {
        "TOP": dataset_pb2.LaserName.TOP,
        "FRONT": dataset_pb2.LaserName.FRONT,
        "SIDE_LEFT": dataset_pb2.LaserName.SIDE_LEFT,
        "SIDE_RIGHT": dataset_pb2.LaserName.SIDE_RIGHT,
        "REAR": dataset_pb2.LaserName.REAR,
    }

    if requested_lidar not in lidar_mapping:
        raise ValueError(
            f"Unknown LiDAR: {requested_lidar}"
        )

    requested_lidar_enum = lidar_mapping[
        requested_lidar
    ]

    frame = load_frame(
        tfrecord_path,
        args.frame_index,
    )

    matching_labels = [
        label
        for label in frame.laser_labels
        if label.id == args.track_id
    ]

    if not matching_labels:
        available = [
            label.id
            for label in frame.laser_labels
        ]

        raise RuntimeError(
            f"Track '{args.track_id}' is not present in "
            f"frame {args.frame_index}. "
            f"Frame contains {len(available)} laser labels."
        )

    label = matching_labels[0]

    print(f"TFRecord: {tfrecord_path}")
    print(f"Context: {frame.context.name}")
    print(f"Frame index: {args.frame_index}")
    print(
        "Timestamp: "
        f"{frame.timestamp_micros}"
    )
    print(f"LiDAR: {requested_lidar}")
    print(f"Return index: {return_index}")

    print_label_details(label)

    (
        range_images,
        camera_projections,
        segmentation_labels,
        range_image_top_pose,
    ) = (
        frame_utils
        .parse_range_image_and_camera_projection(
            frame
        )
    )

    print("\nSegmentation-label availability")
    print("-------------------------------")

    if not segmentation_labels:
        print(
            "No 3D point-level segmentation labels are "
            "stored in this frame."
        )
    else:
        for lidar_enum, returns in sorted(
            segmentation_labels.items(),
            key=lambda item: item[0],
        ):
            print(
                f"{laser_name(lidar_enum)}: "
                f"{len(returns)} labeled return(s)"
            )

            for index, matrix in enumerate(
                returns
            ):
                print(
                    f"  return {index}: "
                    f"shape={list(matrix.shape.dims)}"
                )

    points_per_lidar, _ = (
        frame_utils
        .convert_range_image_to_point_cloud(
            frame,
            range_images,
            camera_projections,
            range_image_top_pose,
            ri_index=return_index,
        )
    )

    labels_per_lidar = (
        convert_segmentation_labels_to_points(
            frame=frame,
            range_images=range_images,
            segmentation_labels=segmentation_labels,
            return_index=return_index,
        )
    )

    calibrations = sorted(
        frame.context.laser_calibrations,
        key=lambda calibration: calibration.name,
    )

    xyz = None
    point_labels = None

    for calibration, lidar_points, lidar_labels in zip(
        calibrations,
        points_per_lidar,
        labels_per_lidar,
    ):
        if (
            calibration.name
            != requested_lidar_enum
        ):
            continue

        xyz = np.asarray(
            lidar_points,
            dtype=np.float64,
        )

        point_labels = np.asarray(
            lidar_labels,
            dtype=np.int32,
        )

        break

    if xyz is None or point_labels is None:
        raise RuntimeError(
            f"Could not extract {requested_lidar}"
        )

    if len(xyz) != len(point_labels):
        raise RuntimeError(
            "Point/segmentation-label count mismatch: "
            f"{len(xyz)} points versus "
            f"{len(point_labels)} labels"
        )

    inside_mask, xyz_object_all = (
        points_inside_label_box(
            xyz_vehicle=xyz,
            label=label,
            margin=args.box_margin,
        )
    )

    box_xyz_vehicle = xyz[inside_mask]
    box_xyz_object = (
        xyz_object_all[inside_mask]
    )
    box_point_labels = (
        point_labels[inside_mask]
    )

    instance_ids = (
        box_point_labels[:, 0]
    )
    semantic_ids = (
        box_point_labels[:, 1]
    )

    print("\nPoint extraction comparison")
    print("---------------------------")
    print(
        f"Selected LiDAR points: {len(xyz):,}"
    )
    print(
        f"Geometric points inside box: "
        f"{inside_mask.sum():,}"
    )
    print(
        "Waymo label num_lidar_points_in_box: "
        f"{label.num_lidar_points_in_box}"
    )

    if hasattr(
        label,
        "num_top_lidar_points_in_box",
    ):
        print(
            "Waymo label "
            "num_top_lidar_points_in_box: "
            f"{label.num_top_lidar_points_in_box}"
        )

    print("\nSemantic classes inside the box")
    print("-------------------------------")

    semantic_counts = Counter(
        int(value)
        for value in semantic_ids
    )

    for value, count in (
        semantic_counts.most_common()
    ):
        print(
            f"{value:2d} "
            f"{semantic_type_name(value):28s} "
            f"{count:6d}"
        )

    print("\nInstance IDs inside the box")
    print("---------------------------")

    instance_counts = Counter(
        int(value)
        for value in instance_ids
    )

    for value, count in (
        instance_counts.most_common(20)
    ):
        print(
            f"instance_id={value:8d} "
            f"points={count:6d}"
        )

    nonzero_instances = [
        (value, count)
        for value, count in (
            instance_counts.most_common()
        )
        if value != 0
    ]

    if nonzero_instances:
        dominant_instance_id = (
            nonzero_instances[0][0]
        )

        dominant_instance_mask = (
            inside_mask.copy()
        )

        dominant_instance_mask[
            inside_mask
        ] = (
            instance_ids
            == dominant_instance_id
        )

        print(
            "\nDominant nonzero instance ID "
            f"inside box: {dominant_instance_id}"
        )
        print(
            "Points assigned to dominant instance: "
            f"{dominant_instance_mask.sum():,}"
        )
    else:
        dominant_instance_id = None
        dominant_instance_mask = np.zeros(
            len(xyz),
            dtype=bool,
        )

        print(
            "\nNo nonzero point-level instance ID "
            "was found inside this box."
        )

    road_class = (
        segmentation_pb2
        .Segmentation.Type.TYPE_ROAD
    )

    road_inside_box = int(
        np.sum(
            semantic_ids == road_class
        )
    )

    print(
        "\nRoad-labeled points inside box: "
        f"{road_inside_box:,}"
    )

    if args.output is not None:
        args.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.savez_compressed(
            args.output,
            xyz_vehicle=xyz.astype(
                np.float32
            ),
            point_instance_id=(
                point_labels[:, 0]
            ),
            point_semantic_class=(
                point_labels[:, 1]
            ),
            box_mask=inside_mask,
            xyz_box_vehicle=(
                box_xyz_vehicle.astype(
                    np.float32
                )
            ),
            xyz_box_object=(
                box_xyz_object.astype(
                    np.float32
                )
            ),
            box_instance_id=instance_ids,
            box_semantic_class=semantic_ids,
            dominant_instance_id=(
                -1
                if dominant_instance_id is None
                else dominant_instance_id
            ),
            dominant_instance_mask=(
                dominant_instance_mask
            ),
        )

        print(
            f"\nSaved debug data: {args.output}"
        )


if __name__ == "__main__":
    main()