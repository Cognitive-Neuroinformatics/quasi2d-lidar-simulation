import argparse
import glob
import os

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
        help=(
            "Directory containing frame_000.npz, "
            "frame_001.npz, ..."
        ),
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
        help="File used to save/load the Open3D camera.",
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
            f"No frame_*.npz files found in: "
            f"{args.input_dir}"
        )

    os.makedirs(
        args.save_dir,
        exist_ok=True,
    )

    print(f"Found {len(frame_files)} frames")
    print()
    print("Controls:")
    print("  N / Right arrow : next frame")
    print("  B / Left arrow  : previous frame")
    print("  C               : save current camera")
    print("  L               : load saved camera")
    print("  S               : save current frame screenshot")
    print("  Q               : quit")
    print()

    # ---------------------------------------------------------
    # Initial frame
    # ---------------------------------------------------------

    current_idx = 0

    xyz = load_frame(
        frame_files[current_idx]
    )

    print(
        f"Initial frame has {len(xyz):,} points"
    )

    # ---------------------------------------------------------
    # Point cloud
    # ---------------------------------------------------------

    pcd = o3d.geometry.PointCloud()

    pcd.points = (
        o3d.utility.Vector3dVector(
            xyz
        )
    )

    # ---------------------------------------------------------
    # Sensor coordinate frame
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

    vis = (
        o3d.visualization.VisualizerWithKeyCallback()
    )

    vis.create_window(
        window_name=(
            "Scala2 reconstructed raycast sequence"
        ),
        width=1600,
        height=900,
    )

    vis.add_geometry(
        pcd,
        reset_bounding_box=True,
    )

    vis.add_geometry(
        coordinate_frame,
        reset_bounding_box=False,
    )

    render_option = (
        vis.get_render_option()
    )

    render_option.point_size = (
        args.point_size
    )

    render_option.point_color_option = (
        o3d.visualization.PointColorOption.ZCoordinate
    )

    # ---------------------------------------------------------
    # Frame update
    # ---------------------------------------------------------

    def update_frame(new_idx):
        nonlocal current_idx

        new_idx = max(
            0,
            min(
                new_idx,
                len(frame_files) - 1,
            ),
        )

        if new_idx == current_idx:
            return False

        current_idx = new_idx

        path = frame_files[
            current_idx
        ]

        xyz_new = load_frame(
            path
        )

        pcd.points = (
            o3d.utility.Vector3dVector(
                xyz_new
            )
        )

        vis.update_geometry(
            pcd
        )

        print(
            f"Frame {current_idx:03d} | "
            f"{os.path.basename(path)} | "
            f"{len(xyz_new):,} points"
        )

        # Important:
        # no reset_bounding_box here.
        # Current camera remains unchanged.

        return False

    # ---------------------------------------------------------
    # Camera save
    # ---------------------------------------------------------

    def save_camera(vis_obj):

        view_control = (
            vis_obj.get_view_control()
        )

        camera_params = (
            view_control
            .convert_to_pinhole_camera_parameters()
        )

        success = (
            o3d.io.write_pinhole_camera_parameters(
                args.camera_file,
                camera_params,
            )
        )

        if success:
            print()
            print(
                "Saved camera:"
            )
            print(
                args.camera_file
            )

            print(
                "Camera extrinsic:"
            )
            print(
                camera_params.extrinsic
            )

        else:
            print(
                "Failed to save camera."
            )

        return False

    # ---------------------------------------------------------
    # Camera load
    # ---------------------------------------------------------

    def load_camera(vis_obj):

        if not os.path.exists(
            args.camera_file
        ):
            print(
                f"Camera file does not exist: "
                f"{args.camera_file}"
            )
            return False

        camera_params = (
            o3d.io.read_pinhole_camera_parameters(
                args.camera_file
            )
        )

        view_control = (
            vis_obj.get_view_control()
        )

        view_control.convert_from_pinhole_camera_parameters(
            camera_params,
            allow_arbitrary=True,
        )

        vis_obj.poll_events()
        vis_obj.update_renderer()

        print(
            f"Loaded camera: "
            f"{args.camera_file}"
        )

        return False

    # ---------------------------------------------------------
    # Save current screenshot
    # ---------------------------------------------------------

    def save_current_frame(vis_obj):

        output_path = os.path.join(
            args.save_dir,
            f"frame_{current_idx:03d}.png",
        )

        vis_obj.poll_events()
        vis_obj.update_renderer()

        success = (
            vis_obj.capture_screen_image(
                output_path,
                do_render=True,
            )
        )

        print(
            f"Saved screenshot: "
            f"{output_path}"
        )

        return False

    # ---------------------------------------------------------
    # Navigation callbacks
    # ---------------------------------------------------------

    def next_frame(vis_obj):
        return update_frame(
            current_idx + 1
        )

    def previous_frame(vis_obj):
        return update_frame(
            current_idx - 1
        )

    def quit_viewer(vis_obj):
        vis_obj.close()
        return False

    # ---------------------------------------------------------
    # Keyboard callbacks
    # ---------------------------------------------------------

    # N
    vis.register_key_callback(
        ord("N"),
        next_frame,
    )

    # B
    vis.register_key_callback(
        ord("B"),
        previous_frame,
    )

    # Right arrow
    vis.register_key_callback(
        262,
        next_frame,
    )

    # Left arrow
    vis.register_key_callback(
        263,
        previous_frame,
    )

    # C
    vis.register_key_callback(
        ord("C"),
        save_camera,
    )

    # L
    vis.register_key_callback(
        ord("L"),
        load_camera,
    )

    # S
    vis.register_key_callback(
        ord("S"),
        save_current_frame,
    )

    # Q
    vis.register_key_callback(
        ord("Q"),
        quit_viewer,
    )

    print(
        f"Frame {current_idx:03d} | "
        f"{os.path.basename(frame_files[current_idx])} | "
        f"{len(xyz):,} points"
    )

    # ---------------------------------------------------------
    # Run
    # ---------------------------------------------------------

    vis.run()

    vis.destroy_window()


if __name__ == "__main__":
    main()