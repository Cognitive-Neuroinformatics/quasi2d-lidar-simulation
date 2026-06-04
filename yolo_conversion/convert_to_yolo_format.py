#!/usr/bin/env python3
"""
Prepare a YOLO OBB dataset from BEV images and OpenPCDet-format labels.

Input OpenPCDet label format:
    x y z length width height yaw class_name

Example:
    20.080720 -4.460201 0.926611 1.144897 1.002529 1.760000 0.029969 Pedestrian

Output YOLO OBB label format:
    class_id x1 y1 x2 y2 x3 y3 x4 y4

The four OBB corner coordinates are projected into the BEV image plane and
normalized by image width and height.
"""

import argparse
import logging
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml


CLASS_MAPPING = {
    "Vehicle": 0,
    "Pedestrian": 1,
    "Cyclist": 2,
}

VALID_SENSORS = {
    "front_left",
    "front_center",
    "front_right",
    "rear_left",
    "rear_center",
    "rear_right",
}


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def remove_if_empty(path: Path) -> None:
    if path.exists() and path.is_dir() and not any(path.iterdir()):
        path.rmdir()


def load_bev_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    bev = cfg["bev"]

    return {
        "img_width": int(bev["x_size"]),
        "img_height": int(bev["y_size"]),
        "resolution": float(bev["voxel_size"]),
    }


def parse_opencdet_label(line: str):
    parts = line.strip().split()

    if len(parts) != 8:
        return None

    x, y, _, length, width, _, yaw = map(float, parts[:7])
    class_name = parts[7]

    if class_name not in CLASS_MAPPING:
        return None

    return {
        "class_id": CLASS_MAPPING[class_name],
        "x": x,
        "y": y,
        "length": length,
        "width": width,
        "yaw": yaw,
    }


def calculate_obb_corners(
    x: float,
    y: float,
    length: float,
    width: float,
    yaw: float,
) -> np.ndarray:
    half_l = length / 2.0
    half_w = width / 2.0

    corners_local = np.array(
        [
            [half_l, half_w],
            [-half_l, half_w],
            [-half_l, -half_w],
            [half_l, -half_w],
        ],
        dtype=np.float32,
    )

    rotation = np.array(
        [
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw), np.cos(yaw)],
        ],
        dtype=np.float32,
    )

    corners = corners_local @ rotation.T
    corners[:, 0] += x
    corners[:, 1] += y

    return corners


def point_to_bev_pixel(
    point: np.ndarray,
    img_height: int,
    resolution: float,
) -> tuple[float, float]:
    """
    Convert sensor-frame BEV coordinates to image pixel coordinates.

    Assumption:
        x forward  -> image x-axis
        y lateral  -> centered image y-axis
    """
    x, y = point

    pixel_x = x / resolution
    pixel_y = img_height / 2.0 - y / resolution

    return pixel_x, pixel_y


def convert_label_file(
    input_label_path: Path,
    output_label_path: Path,
    img_width: int,
    img_height: int,
    resolution: float,
) -> list[str]:
    yolo_lines = []

    with open(input_label_path, "r") as f:
        for line in f:
            parsed = parse_opencdet_label(line)

            if parsed is None:
                continue

            corners = calculate_obb_corners(
                x=parsed["x"],
                y=parsed["y"],
                length=parsed["length"],
                width=parsed["width"],
                yaw=parsed["yaw"],
            )

            normalized = []

            for corner in corners:
                px, py = point_to_bev_pixel(corner, img_height, resolution)

                nx = px / img_width
                ny = py / img_height

                normalized.extend([nx, ny])

            yolo_line = f"{parsed['class_id']} " + " ".join(
                f"{value:.6f}" for value in normalized
            )

            yolo_lines.append(yolo_line)

    if not yolo_lines:
        return []

    output_label_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_label_path, "w") as f:
        f.write("\n".join(yolo_lines) + "\n")

    return yolo_lines

def order_points_clockwise(points):
    points = np.array(points, dtype=np.float32)

    center = points.mean(axis=0)

    angles = np.arctan2(
        points[:, 1] - center[1],
        points[:, 0] - center[0],
    )

    ordered = points[np.argsort(angles)]
    return ordered.astype(np.int32)

def draw_obb_on_image(
    image_path: Path,
    output_path: Path,
    yolo_lines: list[str],
    img_width: int,
    img_height: int,
) -> None:
    image = cv2.imread(str(image_path))

    if image is None:
        logging.warning("Could not read image for visualization: %s", image_path)
        return

    for line in yolo_lines:
        values = line.strip().split()

        if len(values) != 9:
            continue

        class_id = int(values[0])
        coords = list(map(float, values[1:]))

        points = []

        for i in range(0, len(coords), 2):
            px = int(coords[i] * img_width)
            py = int(coords[i + 1] * img_height)
            points.append([px, py])

        points = order_points_clockwise(points)
        points = np.array(points, dtype=np.int32).reshape((-1, 1, 2))

        cv2.polylines(
            image,
            [points],
            isClosed=True,
            color=(0, 255, 0),
            thickness=2,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def create_split_file(output_root: Path, split: str) -> None:
    images_dir = output_root / "images" / split
    split_file = output_root / f"{split}.txt"

    with open(split_file, "w") as f:
        for image_path in sorted(images_dir.glob("*.png")):
            f.write(str(image_path.resolve()) + "\n")


def get_sensors(split_dir: Path, selected_sensors: list[str]) -> list[str]:
    if selected_sensors == ["all"]:
        sensors = sorted(
            [
                path.name
                for path in split_dir.iterdir()
                if path.is_dir() and path.name in VALID_SENSORS
            ]
        )

        logging.info("Detected sensors: %s", ", ".join(sensors))
        return sensors

    invalid = set(selected_sensors) - VALID_SENSORS

    if invalid:
        raise ValueError(
            f"Unknown sensor(s): {sorted(invalid)}. "
            f"Valid sensors are: {sorted(VALID_SENSORS)}"
        )

    return selected_sensors


def process_sensor(
    sensor_dir: Path,
    output_root: Path,
    split: str,
    image_dir_name: str,
    img_width: int,
    img_height: int,
    resolution: float,
    start_index: int,
    save_debug_images: bool,
    cleanup_source: bool,
) -> int:
    labels_dir = sensor_dir / "labels"
    images_dir = sensor_dir / image_dir_name

    if not labels_dir.exists():
        logging.warning("Missing labels directory: %s", labels_dir)
        return start_index

    if not images_dir.exists():
        logging.warning("Missing image directory: %s", images_dir)
        return start_index

    output_images_dir = output_root / "images" / split
    output_labels_dir = output_root / "labels" / split

    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    sample_index = start_index

    for label_path in sorted(labels_dir.glob("*.txt")):
        image_path = images_dir / f"{label_path.stem}.png"

        if not image_path.exists():
            logging.warning("No matching image found for label: %s", label_path.name)
            continue

        output_image_path = output_images_dir / f"{sample_index:07d}.png"
        output_label_path = output_labels_dir / f"{sample_index:07d}.txt"

        yolo_lines = convert_label_file(
            input_label_path=label_path,
            output_label_path=output_label_path,
            img_width=img_width,
            img_height=img_height,
            resolution=resolution,
        )

        if not yolo_lines:
            logging.warning("Skipping label file without valid boxes: %s", label_path)
            continue

        shutil.copy2(image_path, output_image_path)

        if save_debug_images:
            debug_image_path = (
                output_root / "debug_images" / split / f"{sample_index:07d}.png"
            )

            draw_obb_on_image(
                image_path=output_image_path,
                output_path=debug_image_path,
                yolo_lines=yolo_lines,
                img_width=img_width,
                img_height=img_height,
            )

        if cleanup_source:
            image_path.unlink()
            label_path.unlink()

        sample_index += 1

    if cleanup_source:
        remove_if_empty(labels_dir)
        remove_if_empty(images_dir)

    return sample_index


def process_dataset(args) -> None:
    bev_cfg = load_bev_config(args.config)

    image_dir_name = f"images_{args.transform_mode}"
    output_root = args.output_dir / f"yolo_obb_{args.transform_mode}"

    logging.info("Input directory: %s", args.input_dir)
    logging.info("Output directory: %s", output_root)
    logging.info("Image directory name: %s", image_dir_name)
    logging.info("BEV size: %dx%d", bev_cfg["img_width"], bev_cfg["img_height"])
    logging.info("Resolution: %.3f m/pixel", bev_cfg["resolution"])

    if args.cleanup_source:
        logging.warning(
            "cleanup_source is enabled: source images and source labels will be deleted after successful conversion."
        )

    for split in args.splits:
        split_dir = args.input_dir / split

        if not split_dir.exists():
            logging.warning("Split directory does not exist: %s", split_dir)
            continue

        sample_index = 0
        sensors = get_sensors(split_dir, args.sensors)

        for sensor in sensors:
            sensor_dir = split_dir / sensor

            if not sensor_dir.exists():
                logging.warning("Sensor directory does not exist: %s", sensor_dir)
                continue

            logging.info("Processing split=%s sensor=%s", split, sensor)

            sample_index = process_sensor(
                sensor_dir=sensor_dir,
                output_root=output_root,
                split=split,
                image_dir_name=image_dir_name,
                img_width=bev_cfg["img_width"],
                img_height=bev_cfg["img_height"],
                resolution=bev_cfg["resolution"],
                start_index=sample_index,
                save_debug_images=args.save_debug_images,
                cleanup_source=args.cleanup_source,
            )

        create_split_file(output_root, split)
        logging.info("Finished split=%s with %d samples", split, sample_index)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert BEV images and OpenPCDet labels into YOLO OBB format."
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to parameters.yaml containing BEV settings.",
    )

    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Input dataset root containing training/validation sensor folders.",
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory for the YOLO OBB dataset.",
    )

    parser.add_argument(
        "--transform_mode",
        type=str,
        default="overlap",
        choices=["height", "overlap", "overlap_height"],
        help="BEV image type to export.",
    )

    parser.add_argument(
        "--sensors",
        nargs="+",
        default=["all"],
        help="Sensors to process, e.g. front_center or all.",
    )

    parser.add_argument(
        "--splits",
        nargs="+",
        default=["training", "validation"],
        help="Dataset splits to process.",
    )

    parser.add_argument(
        "--save_debug_images",
        action="store_true",
        help="Save BEV images with YOLO OBB bounding boxes drawn on top.",
    )

    parser.add_argument(
        "--cleanup_source",
        action="store_true",
        help=(
            "Delete source images and source labels after successful conversion. "
            "Default behavior keeps the original files."
        ),
    )

    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    process_dataset(args)


if __name__ == "__main__":
    main()
    
    
    
# without cleanup

# python prepare_yolo_obb_bev_dataset.py \
#     --config config/parameters.yaml \
#     --input_dir /data/simulated_data \
#     --output_dir /data/yolo_dataset \
#     --transform_mode overlap \
#     --sensors all \
#     --splits training validation \
#     --save_debug_images



# with cleanup

# python convert_to_yolo_format.py \
#     --config /home/samanti/git_repos/object_detection_dl/quasi2d-lidar-simulation/config/parameters.yaml \
#     --input_dir /data/simulated_data/Passat/overlap_height \
#     --output_dir /data/simulated_data/Passat/yolo_dataset \
#     --transform_mode overlap_height \
#     --sensors all \
#     --splits training validation \
#     --cleanup_source