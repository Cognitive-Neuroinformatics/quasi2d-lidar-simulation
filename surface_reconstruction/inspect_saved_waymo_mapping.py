#!/usr/bin/env python3

"""
Interactive mapping between a saved Waymo range image and point cloud.

Expected files inside --data_dir
--------------------------------
range_image.npy
cartesian_range_image.npy
valid_range_image_indices.npy
point_cloud_xyz.npy

Optional files
--------------
point_cloud_polar_features.npy
point_cloud_xyz.ply
beam_inclinations_deg.npy
lidar_extrinsic.npy

Modes
-----
image_to_3d
    Click a range-image pixel. Its corresponding XYZ point is highlighted
    with a red sphere in Open3D.

3d_to_image
    Select points in Open3D. Their original range-image pixels are then
    highlighted in Matplotlib.

The script uses the numerical .npy files, not the PNG visualizations.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


@dataclass
class WaymoSavedData:
    range_image: np.ndarray
    cartesian_range_image: np.ndarray
    valid_mask: np.ndarray
    valid_rows: np.ndarray
    valid_columns: np.ndarray
    point_cloud_xyz: np.ndarray
    point_cloud_features: Optional[np.ndarray]
    beam_inclinations_deg: Optional[np.ndarray]
    lidar_extrinsic: Optional[np.ndarray]


def require_file(directory: Path, filename: str) -> Path:
    path = directory / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    return path


def load_optional_npy(
    directory: Path,
    filename: str,
) -> Optional[np.ndarray]:
    path = directory / filename

    if not path.exists():
        print(f"Optional file not found: {filename}")
        return None

    array = np.load(path)
    print(f"Loaded {filename}: {array.shape}")

    return array


def interpret_valid_indices(
    indices: np.ndarray,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert valid_range_image_indices.npy into row and column arrays.

    Supported forms:
        [N, 2]  -> each row is [row, column]
        [2, N]  -> first row contains rows, second row contains columns
        [N]     -> flattened indices into an H x W array
    """

    indices = np.asarray(indices)

    if indices.ndim == 2 and indices.shape[1] == 2:
        rows = indices[:, 0]
        columns = indices[:, 1]

    elif indices.ndim == 2 and indices.shape[0] == 2:
        rows = indices[0]
        columns = indices[1]

    elif indices.ndim == 1:
        rows, columns = np.unravel_index(
            indices.astype(np.int64),
            (height, width),
        )

    else:
        raise ValueError(
            "Unsupported valid_range_image_indices.npy shape: "
            f"{indices.shape}\n"
            "Expected [N, 2], [2, N], or flattened [N]."
        )

    rows = np.asarray(rows, dtype=np.int64)
    columns = np.asarray(columns, dtype=np.int64)

    if np.any(rows < 0) or np.any(rows >= height):
        raise ValueError("Some saved row indices are outside the range image.")

    if np.any(columns < 0) or np.any(columns >= width):
        raise ValueError(
            "Some saved column indices are outside the range image."
        )

    return rows, columns


def load_saved_data(data_dir: Path) -> WaymoSavedData:
    range_image = np.load(
        require_file(data_dir, "range_image.npy")
    )

    cartesian_range_image = np.load(
        require_file(data_dir, "cartesian_range_image.npy")
    )

    point_cloud_xyz = np.load(
        require_file(data_dir, "point_cloud_xyz.npy")
    )

    saved_indices = np.load(
        require_file(data_dir, "valid_range_image_indices.npy")
    )

    if range_image.ndim != 3:
        raise ValueError(
            "range_image.npy should have shape [H, W, C], "
            f"but received {range_image.shape}"
        )

    height, width, channels = range_image.shape

    if channels < 3:
        raise ValueError(
            "range_image.npy should contain at least range, intensity, "
            "and elongation channels."
        )

    if cartesian_range_image.shape != (height, width, 3):
        raise ValueError(
            "cartesian_range_image.npy should have shape "
            f"[{height}, {width}, 3], but received "
            f"{cartesian_range_image.shape}"
        )

    if point_cloud_xyz.ndim != 2 or point_cloud_xyz.shape[1] != 3:
        raise ValueError(
            "point_cloud_xyz.npy should have shape [N, 3], "
            f"but received {point_cloud_xyz.shape}"
        )

    valid_rows, valid_columns = interpret_valid_indices(
        saved_indices,
        height,
        width,
    )

    valid_mask = np.zeros((height, width), dtype=bool)
    valid_mask[valid_rows, valid_columns] = True

    # Also check the range channel itself.
    range_valid_mask = range_image[..., 0] > 0

    disagreement = np.count_nonzero(valid_mask != range_valid_mask)

    if disagreement > 0:
        print(
            f"Warning: saved valid indices and range > 0 disagree at "
            f"{disagreement} pixels."
        )
        print(
            "The script will use valid_range_image_indices.npy because it "
            "should preserve the exact saved point-cloud ordering."
        )

    if len(valid_rows) != len(point_cloud_xyz):
        raise ValueError(
            "Mapping length mismatch:\n"
            f"valid indices:    {len(valid_rows)}\n"
            f"point_cloud_xyz:  {len(point_cloud_xyz)}\n"
            "\nThe valid indices must correspond one-to-one with the saved "
            "point cloud."
        )

    point_cloud_features = load_optional_npy(
        data_dir,
        "point_cloud_polar_features.npy",
    )

    if (
        point_cloud_features is not None
        and len(point_cloud_features) != len(point_cloud_xyz)
    ):
        print(
            "Warning: point_cloud_polar_features.npy does not have the same "
            "number of entries as point_cloud_xyz.npy. It will not be used "
            "for point-level lookup."
        )
        point_cloud_features = None

    beam_inclinations_deg = load_optional_npy(
        data_dir,
        "beam_inclinations_deg.npy",
    )

    lidar_extrinsic = load_optional_npy(
        data_dir,
        "lidar_extrinsic.npy",
    )

    verify_mapping(
        cartesian_range_image=cartesian_range_image,
        point_cloud_xyz=point_cloud_xyz,
        valid_rows=valid_rows,
        valid_columns=valid_columns,
    )

    print("\nLoaded dataset")
    print("--------------")
    print(f"Range image:           {range_image.shape}")
    print(f"Cartesian image:       {cartesian_range_image.shape}")
    print(f"Point cloud:           {point_cloud_xyz.shape}")
    print(f"Valid mapped points:   {len(valid_rows)}")
    print(f"Valid percentage:      {100 * len(valid_rows)/(height*width):.2f}%")

    return WaymoSavedData(
        range_image=range_image,
        cartesian_range_image=cartesian_range_image,
        valid_mask=valid_mask,
        valid_rows=valid_rows,
        valid_columns=valid_columns,
        point_cloud_xyz=point_cloud_xyz,
        point_cloud_features=point_cloud_features,
        beam_inclinations_deg=beam_inclinations_deg,
        lidar_extrinsic=lidar_extrinsic,
    )


def verify_mapping(
    cartesian_range_image: np.ndarray,
    point_cloud_xyz: np.ndarray,
    valid_rows: np.ndarray,
    valid_columns: np.ndarray,
) -> None:
    """
    Check whether point_cloud_xyz[i] matches
    cartesian_range_image[row_i, column_i].
    """

    reconstructed_xyz = cartesian_range_image[
        valid_rows,
        valid_columns,
    ]

    differences = np.linalg.norm(
        reconstructed_xyz - point_cloud_xyz,
        axis=1,
    )

    print("\nMapping verification")
    print("--------------------")
    print(f"Mean XYZ difference: {differences.mean():.9f} m")
    print(f"Max XYZ difference:  {differences.max():.9f} m")

    if differences.max() < 1e-4:
        print("Mapping is consistent.")
    else:
        print(
            "Warning: point-cloud ordering may not exactly match the saved "
            "valid indices."
        )


def normalize_values(
    values: np.ndarray,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)

    low = np.percentile(values, lower_percentile)
    high = np.percentile(values, upper_percentile)

    if high <= low:
        return np.zeros_like(values)

    normalized = (values - low) / (high - low)

    return np.clip(normalized, 0.0, 1.0)


def point_colors(
    data: WaymoSavedData,
    color_by: str,
) -> np.ndarray:
    if color_by == "range":
        values = data.range_image[
            data.valid_rows,
            data.valid_columns,
            0,
        ]
        normalized = normalize_values(values)

    elif color_by == "intensity":
        values = data.range_image[
            data.valid_rows,
            data.valid_columns,
            1,
        ]

        # Match the visualization idea used in your saved PNG.
        values = np.arctan(values)
        normalized = normalize_values(values)

    elif color_by == "elongation":
        values = data.range_image[
            data.valid_rows,
            data.valid_columns,
            2,
        ]
        normalized = normalize_values(values)

    elif color_by == "height":
        values = data.point_cloud_xyz[:, 2]
        normalized = normalize_values(values)

    else:
        raise ValueError(f"Unsupported color mode: {color_by}")

    return plt.get_cmap("turbo")(normalized)[:, :3]


def create_point_cloud(
    data: WaymoSavedData,
    color_by: str,
) -> o3d.geometry.PointCloud:
    point_cloud = o3d.geometry.PointCloud()

    point_cloud.points = o3d.utility.Vector3dVector(
        data.point_cloud_xyz.astype(np.float64)
    )

    point_cloud.colors = o3d.utility.Vector3dVector(
        point_colors(data, color_by)
    )

    return point_cloud


def create_range_display(
    data: WaymoSavedData,
    channel: str,
) -> tuple[np.ndarray, str]:
    display = np.full(
        data.valid_mask.shape,
        np.nan,
        dtype=np.float64,
    )

    if channel == "range":
        raw = data.range_image[..., 0]
        display[data.valid_mask] = np.log1p(raw[data.valid_mask])
        label = "log(1 + range in metres)"

    elif channel == "intensity":
        raw = data.range_image[..., 1]
        display[data.valid_mask] = np.arctan(raw[data.valid_mask])
        label = "arctan(intensity)"

    elif channel == "elongation":
        raw = data.range_image[..., 2]
        display[data.valid_mask] = raw[data.valid_mask]
        label = "elongation"

    elif channel == "valid":
        display = data.valid_mask.astype(np.float64)
        label = "valid return"

    else:
        raise ValueError(f"Unsupported image channel: {channel}")

    return display, label


def find_point_index_for_pixel(
    row: int,
    column: int,
    pixel_to_point: np.ndarray,
) -> int:
    return int(pixel_to_point[row, column])


def build_pixel_to_point_map(
    data: WaymoSavedData,
) -> np.ndarray:
    height, width = data.valid_mask.shape

    mapping = np.full(
        (height, width),
        -1,
        dtype=np.int64,
    )

    mapping[
        data.valid_rows,
        data.valid_columns,
    ] = np.arange(len(data.point_cloud_xyz), dtype=np.int64)

    return mapping


def print_selection(
    data: WaymoSavedData,
    row: int,
    column: int,
    point_index: int,
) -> None:
    values = data.range_image[row, column]
    xyz = data.point_cloud_xyz[point_index]

    print("\n" + "=" * 72)
    print(f"Point-cloud index: {point_index}")
    print(f"Range-image pixel: row={row}, column={column}")
    print(f"Range:              {values[0]:.6f} m")
    print(f"Intensity:          {values[1]:.6f}")
    print(f"Elongation:         {values[2]:.6f}")

    if values.shape[0] > 3:
        print(f"Channel 3:          {values[3]:.6f}")

    if data.beam_inclinations_deg is not None:
        inclinations = np.ravel(data.beam_inclinations_deg)

        if row < len(inclinations):
            print(
                f"Beam inclination:   "
                f"{inclinations[row]:.6f} degrees"
            )

    print(
        "XYZ:                "
        f"[{xyz[0]:.6f}, {xyz[1]:.6f}, {xyz[2]:.6f}] m"
    )

    print(
        f"Horizontal distance: "
        f"{np.hypot(xyz[0], xyz[1]):.6f} m"
    )

    print(
        f"Azimuth from XYZ:    "
        f"{np.degrees(np.arctan2(xyz[1], xyz[0])):.6f} degrees"
    )

    print("=" * 72)


def create_marker(radius: float) -> o3d.geometry.TriangleMesh:
    marker = o3d.geometry.TriangleMesh.create_sphere(
        radius=radius,
        resolution=0.3,
    )
    marker.compute_vertex_normals()
    marker.paint_uniform_color([1.0, 0.0, 0.0])

    return marker


def run_image_to_3d(
    data: WaymoSavedData,
    image_channel: str,
    point_color: str,
    marker_radius: float,
) -> None:
    point_cloud = create_point_cloud(data, point_color)
    pixel_to_point = build_pixel_to_point_map(data)

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(
        window_name="Waymo pixel-to-point mapping",
        width=1400,
        height=900,
    )

    visualizer.add_geometry(point_cloud)

    coordinate_frame = (
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0)
    )
    visualizer.add_geometry(coordinate_frame)

    marker = create_marker(marker_radius)
    marker_position = np.zeros(3, dtype=np.float64)
    marker.translate(marker_position)
    visualizer.add_geometry(marker)

    display, colorbar_label = create_range_display(
        data,
        image_channel,
    )

    figure, axis = plt.subplots(figsize=(18, 6))

    image_artist = axis.imshow(
        display,
        origin="upper",
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
    )

    selected_artist, = axis.plot(
        [],
        [],
        marker="o",
        markersize=12,
        markerfacecolor="none",
        markeredgecolor="red",
        markeredgewidth=2,
    )

    axis.set_title(
        "Click a valid pixel to highlight its corresponding 3D point"
    )
    axis.set_xlabel("Range-image column / azimuth sample")
    axis.set_ylabel("Range-image row / beam inclination")

    figure.colorbar(
        image_artist,
        ax=axis,
        label=colorbar_label,
    )

    def on_click(event) -> None:
        nonlocal marker_position

        if event.inaxes is not axis:
            return

        if event.xdata is None or event.ydata is None:
            return

        column = int(round(event.xdata))
        row = int(round(event.ydata))

        height, width = data.valid_mask.shape

        if not (0 <= row < height and 0 <= column < width):
            return

        point_index = find_point_index_for_pixel(
            row,
            column,
            pixel_to_point,
        )

        if point_index < 0:
            print(
                f"Pixel row={row}, column={column} is not a valid return."
            )
            return

        selected_xyz = data.point_cloud_xyz[
            point_index
        ].astype(np.float64)

        print_selection(
            data,
            row,
            column,
            point_index,
        )

        marker.translate(selected_xyz - marker_position)
        marker_position = selected_xyz

        visualizer.update_geometry(marker)

        selected_artist.set_data(
            [column],
            [row],
        )

        figure.canvas.draw_idle()

    figure.canvas.mpl_connect(
        "button_press_event",
        on_click,
    )

    plt.show(block=False)

    print("\nPixel-to-point mode")
    print("-------------------")
    print("Click a valid range-image pixel.")
    print("A red sphere will mark its exact point in Open3D.")
    print("Close either window to finish.")

    try:
        while plt.fignum_exists(figure.number):
            if not visualizer.poll_events():
                break

            visualizer.update_renderer()
            plt.pause(0.01)

    except KeyboardInterrupt:
        pass

    finally:
        visualizer.destroy_window()
        plt.close(figure)


def run_3d_to_image(
    data: WaymoSavedData,
    image_channel: str,
    point_color: str,
) -> None:
    point_cloud = create_point_cloud(data, point_color)

    print("\nPoint selection controls")
    print("------------------------")
    print("1. Rotate to the desired viewpoint.")
    print("2. Press Shift + left click to select a point.")
    print("3. Shift + right click undoes a selection.")
    print("4. Press Q when finished.")

    visualizer = o3d.visualization.VisualizerWithEditing()
    visualizer.create_window(
        window_name="Select Waymo point-cloud points",
        width=1400,
        height=900,
    )

    visualizer.add_geometry(point_cloud)
    visualizer.run()

    selected_indices = visualizer.get_picked_points()
    visualizer.destroy_window()

    if not selected_indices:
        print("No points selected.")
        return

    display, colorbar_label = create_range_display(
        data,
        image_channel,
    )

    figure, axis = plt.subplots(figsize=(18, 6))

    image_artist = axis.imshow(
        display,
        origin="upper",
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
    )

    selected_rows = []
    selected_columns = []

    for number, point_index in enumerate(
        selected_indices,
        start=1,
    ):
        point_index = int(point_index)

        row = int(data.valid_rows[point_index])
        column = int(data.valid_columns[point_index])

        selected_rows.append(row)
        selected_columns.append(column)

        print(f"\nSelected point {number}")

        print_selection(
            data,
            row,
            column,
            point_index,
        )

        axis.text(
            column,
            row,
            str(number),
            color="red",
            fontsize=11,
            fontweight="bold",
        )

    axis.scatter(
        selected_columns,
        selected_rows,
        s=110,
        facecolors="none",
        edgecolors="red",
        linewidths=2,
    )

    axis.set_title(
        "Range-image pixels corresponding to selected 3D points"
    )
    axis.set_xlabel("Range-image column / azimuth sample")
    axis.set_ylabel("Range-image row / beam inclination")

    figure.colorbar(
        image_artist,
        ax=axis,
        label=colorbar_label,
    )

    plt.show()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interactively inspect the mapping between saved Waymo "
            "range-image pixels and point-cloud points."
        )
    )

    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help=(
            "Directory containing range_image.npy, "
            "cartesian_range_image.npy, point_cloud_xyz.npy and "
            "valid_range_image_indices.npy."
        ),
    )

    parser.add_argument(
        "--mode",
        choices=["image_to_3d", "3d_to_image"],
        default="image_to_3d",
    )

    parser.add_argument(
        "--image_channel",
        choices=["range", "intensity", "elongation", "valid"],
        default="range",
        help="Channel displayed in the Matplotlib range image.",
    )

    parser.add_argument(
        "--point_color",
        choices=["range", "intensity", "elongation", "height"],
        default="range",
        help="Feature used to color the Open3D point cloud.",
    )

    parser.add_argument(
        "--marker_radius",
        type=float,
        default=0.05,
        help="Radius in metres of the red selected-point marker.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    if not args.data_dir.is_dir():
        raise NotADirectoryError(
            f"Directory does not exist: {args.data_dir}"
        )

    data = load_saved_data(args.data_dir)

    if args.mode == "image_to_3d":
        run_image_to_3d(
            data=data,
            image_channel=args.image_channel,
            point_color=args.point_color,
            marker_radius=args.marker_radius,
        )
    else:
        run_3d_to_image(
            data=data,
            image_channel=args.image_channel,
            point_color=args.point_color,
        )


if __name__ == "__main__":
    main()
    
    
# python inspect_saved_waymo_mapping.py \
#     --data_dir /home/samanti/Documents/Uni_Bremen/PhD/understanding_range_images_waymo/all_lidars/range_image_debug_output/top/return_1 \
#     --mode 3d_to_image \
#     --image_channel range \
#     --point_color range