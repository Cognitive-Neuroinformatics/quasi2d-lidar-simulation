import argparse
import glob
import os
import time

import numpy as np
import open3d as o3d


def load_frame(path):
    """
    Load XYZ points from one saved raycast frame.
    """
    data = np.load(path)

    if "xyz" not in data.files:
        raise KeyError(
            f"'xyz' not found in {path}. "
            f"Available keys: {data.files}"
        )

    return data["xyz"].astype(np.float64)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing frame_000.npz, frame_001.npz, ...",
    )

    parser.add_argument(
        "--point-size",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default="rendered_frames",
        help="Directory for saved PNG screenshots.",
    )

    parser.add_argument(
        "--camera-file",
        type=str,
        default="camera_view.json",
        help="Saved Open3D camera parameters.",
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Find frame files
    # ---------------------------------------------------------

    frame_files = sorted(
        glob.glob(
            os.path.join(
                args.input_dir,
                "frame_*.npz",
            )
        )
    )

    if not frame_files:
        raise RuntimeError(
            f"No frame_*.npz files found in: {args.input_dir}"
        )

    if not os.path.exists(args.camera_file):
        raise FileNotFoundError(
            f"Camera file not found: {args.camera_file}"
        )

    os.makedirs(
        args.save_dir,
        exist_ok=True,
    )

    print(f"Found {len(frame_files)} frames")
    print(f"Camera file: {args.camera_file}")
    print(f"Output directory: {args.save_dir}")

    # ---------------------------------------------------------
    # Initial frame
    # ---------------------------------------------------------

    xyz = load_frame(
        frame_files[0]
    )

    pcd = o3d.geometry.PointCloud()

    pcd.points = (
        o3d.utility.Vector3dVector(
            xyz
        )
    )

    # Blue point cloud
    pcd.paint_uniform_color(
        [0.0, 0.0, 1.0]
    )

    # ---------------------------------------------------------
    # Coordinate frame
    # ---------------------------------------------------------

    coordinate_frame = (
        o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=2.0,
            origin=[0.0, 0.0, 0.0],
        )
    )

    # ---------------------------------------------------------
    # Visualizer
    # ---------------------------------------------------------

    vis = o3d.visualization.Visualizer()

    vis.create_window(
        window_name="Scala2 batch renderer",
        width=1600,
        height=900,
        visible=True,
    )

    vis.add_geometry(
        pcd,
        reset_bounding_box=True,
    )

    vis.add_geometry(
        coordinate_frame,
        reset_bounding_box=False,
    )

    render_option = vis.get_render_option()

    render_option.point_size = (
        args.point_size
    )

    render_option.point_color_option = (
        o3d.visualization.PointColorOption.Color
    )

    # ---------------------------------------------------------
    # Load saved camera
    # ---------------------------------------------------------

    camera_params = (
        o3d.io.read_pinhole_camera_parameters(
            args.camera_file
        )
    )

    view_control = (
        vis.get_view_control()
    )

    view_control.convert_from_pinhole_camera_parameters(
        camera_params,
        allow_arbitrary=True,
    )

    vis.poll_events()
    vis.update_renderer()

    print()
    print("Loaded camera:")
    print(camera_params.extrinsic)
    print()

    # ---------------------------------------------------------
    # Render all frames automatically
    # ---------------------------------------------------------

    for idx, path in enumerate(frame_files):

        xyz = load_frame(
            path
        )

        # Update points
        pcd.points = (
            o3d.utility.Vector3dVector(
                xyz
            )
        )

        # Repaint every new frame blue
        pcd.paint_uniform_color(
            [0.0, 0.0, 1.0]
        )

        vis.update_geometry(
            pcd
        )

        # Reapply camera to guarantee that all frames
        # are rendered from exactly the same viewpoint.
        view_control.convert_from_pinhole_camera_parameters(
            camera_params,
            allow_arbitrary=True,
        )

        # Render updated geometry
        vis.poll_events()
        vis.update_renderer()

        # Small pause lets Open3D finish updating
        time.sleep(0.02)

        output_path = os.path.join(
            args.save_dir,
            f"frame_{idx:03d}.png",
        )

        vis.capture_screen_image(
            output_path,
            do_render=True,
        )

        print(
            f"[{idx:03d}/{len(frame_files)-1:03d}] "
            f"{os.path.basename(path)} | "
            f"{len(xyz):,} points -> "
            f"{output_path}"
        )

    # ---------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------

    vis.destroy_window()

    print()
    print(
        f"Finished rendering {len(frame_files)} frames."
    )


if __name__ == "__main__":
    main()