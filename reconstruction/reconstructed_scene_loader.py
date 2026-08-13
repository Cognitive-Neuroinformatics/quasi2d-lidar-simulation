import os

import numpy as np


class ReconstructedSceneLoader:
    """
    Load the reconstructed static scene and frame-specific dynamic objects.

    Input structure:

        full_scene/
        ├── static_scene.npy
        └── frames/
            ├── frame_000.npz
            ├── frame_001.npz
            └── ...

    static_scene.npy
        XYZ in world coordinates.

    frame_xxx.npz["xyz"]
        Dynamic reconstructed objects already placed in world coordinates.

    frame_xxx.npz["vehicle_to_world"]
        Vehicle/base_link pose for this timestamp.
    """

    def __init__(
        self,
        reconstructed_scene_dir,
    ):
        self.reconstructed_scene_dir = (
            reconstructed_scene_dir
        )

        self.static_path = os.path.join(
            reconstructed_scene_dir,
            "static_scene.npy",
        )

        self.frames_dir = os.path.join(
            reconstructed_scene_dir,
            "frames",
        )

        if not os.path.exists(
            self.static_path
        ):
            raise FileNotFoundError(
                self.static_path
            )

        if not os.path.isdir(
            self.frames_dir
        ):
            raise FileNotFoundError(
                self.frames_dir
            )

        print(
            "[INFO] Loading reconstructed static scene:"
        )

        print(
            self.static_path
        )

        self.static_world = np.load(
            self.static_path
        ).astype(np.float32)

        print(
            "[INFO] Static reconstruction points:",
            len(self.static_world),
        )


    @staticmethod
    def transform_points(
        points_xyz,
        T,
    ):
        """
        Apply 4x4 homogeneous transform to Nx3 XYZ points.
        """

        points_xyz = np.asarray(
            points_xyz,
            dtype=np.float32,
        )

        ones = np.ones(
            (len(points_xyz), 1),
            dtype=np.float32,
        )

        points_h = np.concatenate(
            [
                points_xyz,
                ones,
            ],
            axis=1,
        )

        transformed = (
            points_h
            @ T.T
        )

        return transformed[:, :3]


    def load_frame_world(
        self,
        frame_idx,
    ):
        """
        Return complete reconstructed scene for one timestamp
        in WORLD coordinates.
        """

        frame_path = os.path.join(
            self.frames_dir,
            f"frame_{frame_idx:03d}.npz",
        )

        if not os.path.exists(
            frame_path
        ):
            raise FileNotFoundError(
                frame_path
            )

        data = np.load(
            frame_path
        )

        dynamic_world = (
            data["xyz"]
            .astype(np.float32)
        )

        T_vehicle_to_world = (
            data["vehicle_to_world"]
            .astype(np.float32)
        )

        complete_world = np.concatenate(
            [
                self.static_world,
                dynamic_world,
            ],
            axis=0,
        )

        return (
            complete_world,
            T_vehicle_to_world,
            dynamic_world,
        )


    def load_frame_vehicle(
        self,
        frame_idx,
        point_cloud_range=None,
    ):
        """
        Build complete reconstructed frame in current VEHICLE/base_link frame.

        WORLD
          -> inverse(vehicle_to_world)
          -> VEHICLE

        Optionally crop using:
            [xmin, ymin, zmin, xmax, ymax, zmax]
        """

        (
            complete_world,
            T_vehicle_to_world,
            dynamic_world,
        ) = self.load_frame_world(
            frame_idx
        )

        T_world_to_vehicle = (
            np.linalg.inv(
                T_vehicle_to_world
            )
            .astype(np.float32)
        )

        complete_vehicle = (
            self.transform_points(
                complete_world,
                T_world_to_vehicle,
            )
        )

        if point_cloud_range is not None:

            pc_range = np.asarray(
                point_cloud_range,
                dtype=np.float32,
            )

            if pc_range.shape != (6,):
                raise ValueError(
                    "point_cloud_range must be "
                    "[xmin,ymin,zmin,xmax,ymax,zmax]"
                )

            mask = (
                (
                    complete_vehicle[:, 0]
                    >= pc_range[0]
                )
                &
                (
                    complete_vehicle[:, 0]
                    <= pc_range[3]
                )
                &
                (
                    complete_vehicle[:, 1]
                    >= pc_range[1]
                )
                &
                (
                    complete_vehicle[:, 1]
                    <= pc_range[4]
                )
                &
                (
                    complete_vehicle[:, 2]
                    >= pc_range[2]
                )
                &
                (
                    complete_vehicle[:, 2]
                    <= pc_range[5]
                )
            )

            complete_vehicle = (
                complete_vehicle[
                    mask
                ]
            )

        return {
            "xyz": complete_vehicle,
            "vehicle_to_world": (
                T_vehicle_to_world
            ),
            "world_to_vehicle": (
                T_world_to_vehicle
            ),
            "num_static_world": len(
                self.static_world
            ),
            "num_dynamic_world": len(
                dynamic_world
            ),
        }