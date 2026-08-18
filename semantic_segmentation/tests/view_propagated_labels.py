import argparse
import glob
import os

import numpy as np
import open3d as o3d


CLASS_COLORS = {
    0:  [0.15, 0.15, 0.15],
    1:  [1.0, 0.0, 0.0],       # CAR
    2:  [0.8, 0.2, 0.0],       # TRUCK
    3:  [0.7, 0.0, 0.2],       # BUS
    4:  [0.6, 0.2, 0.2],       # OTHER_VEHICLE
    5:  [1.0, 0.4, 0.0],       # MOTORCYCLIST
    6:  [1.0, 0.7, 0.0],       # BICYCLIST
    7:  [1.0, 0.0, 1.0],       # PEDESTRIAN
    8:  [0.0, 0.0, 1.0],       # SIGN
    9:  [1.0, 1.0, 0.0],       # TRAFFIC_LIGHT
    10: [0.4, 0.4, 1.0],       # POLE
    11: [1.0, 0.5, 0.0],       # CONSTRUCTION_CONE
    12: [0.9, 0.8, 0.1],       # BICYCLE
    13: [0.8, 0.5, 0.0],       # MOTORCYCLE
    14: [0.6, 0.6, 0.6],       # BUILDING
    15: [0.0, 0.7, 0.0],       # VEGETATION
    16: [0.4, 0.25, 0.1],      # TREE_TRUNK
    17: [0.0, 1.0, 1.0],       # CURB
    18: [0.2, 0.2, 0.2],       # ROAD
    19: [1.0, 1.0, 1.0],       # LANE_MARKER
    20: [0.5, 0.4, 0.3],       # OTHER_GROUND
    21: [0.3, 0.8, 0.8],       # WALKABLE
    22: [0.7, 0.7, 0.3],       # SIDEWALK
}


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
# Ground / non-ground definition
# -------------------------------------------------------------------------

GROUND_CLASSES = {
    17,  # CURB
    18,  # ROAD
    19,  # LANE_MARKER
    20,  # OTHER_GROUND
    21,  # WALKABLE
    22,  # SIDEWALK
}


GROUND_VIEW_COLORS = {
    "ground": [0.0, 1.0, 0.0],
    "non_ground": [1.0, 0.0, 0.0],
    "unknown": [0.2, 0.2, 0.2],
}


class SemanticViewer:

    def __init__(
        self,
        files,
        hide_unknown=False,
        view_mode="semantic",
    ):
        self.files = files
        self.hide_unknown = hide_unknown
        self.view_mode = view_mode
        self.index = 0

        self.pcd = o3d.geometry.PointCloud()

    def make_colors(self, semantic):
        """
        Generate colors depending on selected view mode.
        """

        colors = np.zeros(
            (len(semantic), 3),
            dtype=np.float64,
        )

        # -------------------------------------------------------------
        # Standard semantic view
        # -------------------------------------------------------------

        if self.view_mode == "semantic":

            for cls in np.unique(semantic):

                colors[semantic == cls] = CLASS_COLORS.get(
                    int(cls),
                    [1.0, 1.0, 1.0],
                )

        # -------------------------------------------------------------
        # Binary ground / non-ground view
        # -------------------------------------------------------------

        elif self.view_mode == "ground":

            unknown_mask = semantic == 0

            ground_mask = np.isin(
                semantic,
                list(GROUND_CLASSES),
            )

            non_ground_mask = (
                (~ground_mask)
                & (~unknown_mask)
            )

            colors[ground_mask] = (
                GROUND_VIEW_COLORS["ground"]
            )

            colors[non_ground_mask] = (
                GROUND_VIEW_COLORS["non_ground"]
            )

            colors[unknown_mask] = (
                GROUND_VIEW_COLORS["unknown"]
            )

        else:
            raise ValueError(
                f"Unknown view mode: {self.view_mode}"
            )

        return colors

    def print_class_statistics(self, semantic):
        """
        Print useful class statistics depending on view mode.
        """

        if self.view_mode == "semantic":

            classes, counts = np.unique(
                semantic,
                return_counts=True,
            )

            print("\nClasses:")

            for cls, count in zip(classes, counts):

                print(
                    f"  "
                    f"{CLASS_NAMES.get(int(cls), 'UNKNOWN'):20s} "
                    f"{count:8d}"
                )

        elif self.view_mode == "ground":

            unknown_mask = semantic == 0

            ground_mask = np.isin(
                semantic,
                list(GROUND_CLASSES),
            )

            non_ground_mask = (
                (~ground_mask)
                & (~unknown_mask)
            )

            print("\nGround statistics:")

            print(
                f"  Ground       : "
                f"{np.sum(ground_mask):8d}"
            )

            print(
                f"  Non-ground   : "
                f"{np.sum(non_ground_mask):8d}"
            )

            print(
                f"  Unknown      : "
                f"{np.sum(unknown_mask):8d}"
            )

    def load_frame(self):
        """
        Load current NPZ frame and update the Open3D point cloud.
        """

        path = self.files[self.index]

        data = np.load(path)

        xyz = data["xyz"]
        semantic = data["semantic_class"]

        is_gt = bool(data["is_ground_truth"])
        source_frame = int(data["source_frame"])

        # -------------------------------------------------------------
        # Optionally remove unknown points
        # -------------------------------------------------------------

        if self.hide_unknown:

            mask = semantic != 0

            xyz = xyz[mask]
            semantic = semantic[mask]

        # -------------------------------------------------------------
        # Colors
        # -------------------------------------------------------------

        colors = self.make_colors(
            semantic
        )

        # -------------------------------------------------------------
        # Update Open3D point cloud
        # -------------------------------------------------------------

        self.pcd.points = (
            o3d.utility.Vector3dVector(
                xyz.astype(np.float64)
            )
        )

        self.pcd.colors = (
            o3d.utility.Vector3dVector(
                colors
            )
        )

        # -------------------------------------------------------------
        # Print info
        # -------------------------------------------------------------

        frame_name = os.path.basename(path)

        print("\n========================================")
        print(
            f"{frame_name} "
            f"({self.index + 1}/{len(self.files)})"
        )

        if is_gt:

            print("Type      : GROUND TRUTH")

        else:

            print("Type      : PROPAGATED")
            print(
                f"Source    : frame {source_frame:03d}"
            )

        print(
            f"View mode : {self.view_mode}"
        )

        print(
            f"Points    : {len(xyz)}"
        )

        self.print_class_statistics(
            semantic
        )

        print("========================================")

    def next_frame(self, vis):
        """
        Move to next frame.
        """

        if self.index >= len(self.files) - 1:

            print(
                "\nAlready at last frame."
            )

            return False

        self.index += 1

        self.load_frame()

        vis.update_geometry(
            self.pcd
        )

        vis.poll_events()
        vis.update_renderer()

        return False

    def previous_frame(self, vis):
        """
        Move to previous frame.
        """

        if self.index <= 0:

            print(
                "\nAlready at first frame."
            )

            return False

        self.index -= 1

        self.load_frame()

        vis.update_geometry(
            self.pcd
        )

        vis.poll_events()
        vis.update_renderer()

        return False


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        required=True,
        help=(
            "Directory containing "
            "frame_000.npz, frame_001.npz, ..."
        ),
    )

    parser.add_argument(
        "--point-size",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--hide-unknown",
        action="store_true",
        help=(
            "Hide points whose semantic class is 0."
        ),
    )

    parser.add_argument(
        "--view-mode",
        choices=[
            "semantic",
            "ground",
        ],
        default="semantic",
        help=(
            "semantic = original Waymo semantic colors; "
            "ground = binary ground/non-ground visualization."
        ),
    )

    args = parser.parse_args()

    # ---------------------------------------------------------------------
    # Find frames
    # ---------------------------------------------------------------------

    files = sorted(
        glob.glob(
            os.path.join(
                args.input_dir,
                "frame_*.npz",
            )
        )
    )

    if not files:

        raise RuntimeError(
            f"No frame_*.npz files found in "
            f"{args.input_dir}"
        )

    print(
        f"Found {len(files)} frames."
    )

    # ---------------------------------------------------------------------
    # Viewer
    # ---------------------------------------------------------------------

    viewer = SemanticViewer(
        files,
        hide_unknown=args.hide_unknown,
        view_mode=args.view_mode,
    )

    viewer.load_frame()

    # ---------------------------------------------------------------------
    # Persistent Open3D window
    # ---------------------------------------------------------------------

    vis = (
        o3d.visualization
        .VisualizerWithKeyCallback()
    )

    success = vis.create_window(
        window_name="Waymo Semantic Propagation",
        width=1600,
        height=900,
    )

    if not success:

        raise RuntimeError(
            "Could not create Open3D window."
        )

    vis.add_geometry(
        viewer.pcd
    )

    render = vis.get_render_option()

    render.point_size = (
        args.point_size
    )

    render.background_color = np.array(
        [0.05, 0.05, 0.05]
    )

    # ---------------------------------------------------------------------
    # Keyboard controls
    # ---------------------------------------------------------------------

    # N = next frame
    vis.register_key_callback(
        ord("N"),
        viewer.next_frame,
    )

    # P = previous frame
    vis.register_key_callback(
        ord("P"),
        viewer.previous_frame,
    )

    print("\nControls")
    print("--------------------------------")
    print("N : next frame")
    print("P : previous frame")
    print("Q : quit")
    print("--------------------------------")

    if args.view_mode == "semantic":

        print("\nView mode: SEMANTIC")
        print("Each semantic class has its own color.")

    elif args.view_mode == "ground":

        print("\nView mode: GROUND / NON-GROUND")
        print("GREEN : ground")
        print("RED   : non-ground")
        print("GRAY  : unknown")

        print(
            "\nGround classes:",
            sorted(GROUND_CLASSES),
        )

        print(
            " 17 CURB\n"
            " 18 ROAD\n"
            " 19 LANE_MARKER\n"
            " 20 OTHER_GROUND\n"
            " 21 WALKABLE\n"
            " 22 SIDEWALK"
        )

    # ---------------------------------------------------------------------
    # Start
    # ---------------------------------------------------------------------

    vis.run()

    vis.destroy_window()


if __name__ == "__main__":
    main()