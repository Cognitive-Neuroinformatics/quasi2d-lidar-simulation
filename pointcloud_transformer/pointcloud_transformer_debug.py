import os
import csv
import math
import numpy as np
import transformations as tf_transformations
from sklearn.linear_model import RANSACRegressor
from sklearn.cluster import DBSCAN
from datetime import datetime
import copy
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import open3d as o3d
import matplotlib.pyplot as plt
from datetime import datetime


class PointCloudTransformer:
    def __init__(self, voxel_size, ground_removal_method='default'):
        self.voxel_size = voxel_size
        self.ground_removal_method = ground_removal_method
        self.current_transformed_pointcloud = None
        
        # Just for Scala2 sensor - building it once and caching it
        self.az_deg = self.build_scala_azimuths_deg(include_edges=True, include_center_edges=True)  


    @staticmethod
    def scala2_polar_angle_deg(
        azimuth_deg,
        mirror_side,
        apd_group,
        layer,
    ):
        """
        SCALA Gen2 Equation 1.

        Parameters
        ----------
        azimuth_deg : float or np.ndarray
            Azimuthal angle phi in degrees, in the sensor-local frame.
        mirror_side : int
            0 = upward-deflecting mirror, red in Figure 3.
            1 = downward-deflecting mirror, black in Figure 3.
        apd_group : int
            APD group index 0..3, counted bottom to top.
        layer : int
            Layer index 0..3 inside the APD group, bottom to top.

        Returns
        -------
        float or np.ndarray
            Polar/elevation angle theta in degrees.
        """
        if mirror_side not in (0, 1):
            raise ValueError("mirror_side must be 0 or 1")

        if not 0 <= apd_group <= 3:
            raise ValueError("apd_group must be in [0, 3]")

        if not 0 <= layer <= 3:
            raise ValueError("layer must be in [0, 3]")

        phi = np.asarray(azimuth_deg, dtype=np.float64)

        mirror_term = (
            1.512e-8 * phi**3
            - 5.152e-6 * phi**2
            - 1.233e-3 * phi
            + 0.1412
        )

        base_layer_angle = (
            0.6025 * layer
            + 2.564 * apd_group
            - 4.749
        )

        mirror_sign = 1.0 if mirror_side == 0 else -1.0

        return base_layer_angle + mirror_sign * mirror_term
    
    @staticmethod
    def generate_scala2_base_azimuths_deg(
        horizontal_angle_min=-66.5,
        horizontal_angle_max=66.5,
        inner_angle_min=-15.0,
        inner_angle_max=15.0,
        outer_increment=0.25,
        inner_increment=0.125,
    ):
        """
        Generate nominal SCALA Gen2 column azimuths.

        Regions:
        [horizontal_angle_min, inner_angle_min) -> outer increment
        [inner_angle_min, inner_angle_max)      -> inner increment
        [inner_angle_max, horizontal_angle_max] -> outer increment
        """
        left_outer = np.arange(
            horizontal_angle_min,
            inner_angle_min,
            outer_increment,
            dtype=np.float64,
        )

        center = np.arange(
            inner_angle_min,
            inner_angle_max,
            inner_increment,
            dtype=np.float64,
        )

        right_outer = np.arange(
            inner_angle_max,
            horizontal_angle_max + 1e-9,
            outer_increment,
            dtype=np.float64,
        )

        azimuths = np.concatenate(
            (left_outer, center, right_outer)
        )

        # Protect against floating-point duplicates at region boundaries.
        return np.unique(np.round(azimuths, decimals=8))
    
    @staticmethod
    def quaternion_xyzw_to_rotation_matrix(
        qx,
        qy,
        qz,
        qw,
    ):
        """
        Convert an xyzw quaternion into a 3x3 rotation matrix.

        The returned matrix maps vectors from the sensor-local frame
        into the parent/base_link frame, assuming the quaternion is the
        sensor pose orientation expressed in base_link.
        """
        quaternion = np.asarray(
            [qx, qy, qz, qw],
            dtype=np.float64,
        )

        norm = np.linalg.norm(quaternion)

        if norm < 1e-12:
            raise ValueError("Quaternion norm is approximately zero")

        qx, qy, qz, qw = quaternion / norm

        return np.array(
            [
                [
                    1.0 - 2.0 * (qy * qy + qz * qz),
                    2.0 * (qx * qy - qz * qw),
                    2.0 * (qx * qz + qy * qw),
                ],
                [
                    2.0 * (qx * qy + qz * qw),
                    1.0 - 2.0 * (qx * qx + qz * qz),
                    2.0 * (qy * qz - qx * qw),
                ],
                [
                    2.0 * (qx * qz - qy * qw),
                    2.0 * (qy * qz + qx * qw),
                    1.0 - 2.0 * (qx * qx + qy * qy),
                ],
            ],
            dtype=np.float64,
        )

    def calculate_endpoints_vectorized_scala2(
        self,
        start_point_xyz,
        dist_m,
        mirror_side,
        rotation_quaternion_xyzw,
        horizontal_angle_min=-66.5,
        horizontal_angle_max=66.5,
        inner_angle_min=-15.0,
        inner_angle_max=15.0,
        outer_increment_deg=0.25,
        inner_increment_deg=0.125,
        apd_group_azimuth_offset_deg=0.0181,
        apd_group_azimuth_sign=1.0,
    ):
        """
        Generate SCALA Gen2 ray endpoints using:
        - non-uniform horizontal resolution,
        - APD-group azimuth staggering,
        - Equation 1 polar angles,
        - selected mirror side,
        - full sensor quaternion orientation.

        Returns
        -------
        endpoints_vehicle : np.ndarray, shape (N, 3)
            Ray endpoints in the Waymo/base_link coordinate frame.

        metadata : dict[str, np.ndarray]
            Per-ray angular and channel metadata.
        """
        start_point_xyz = np.asarray(
            start_point_xyz,
            dtype=np.float64,
        ).reshape(3)

        qx, qy, qz, qw = rotation_quaternion_xyzw

        rotation_sensor_to_vehicle = (
            self.quaternion_xyzw_to_rotation_matrix(
                qx=qx,
                qy=qy,
                qz=qz,
                qw=qw,
            )
        )

        base_azimuths_deg = (
            self.generate_scala2_base_azimuths_deg(
                horizontal_angle_min=horizontal_angle_min,
                horizontal_angle_max=horizontal_angle_max,
                inner_angle_min=inner_angle_min,
                inner_angle_max=inner_angle_max,
                outer_increment=outer_increment_deg,
                inner_increment=inner_increment_deg,
            )
        )

        directions_sensor = []
        ray_azimuths_deg = []
        ray_elevations_deg = []
        ray_apd_groups = []
        ray_layers = []
        ray_column_ids = []

        for column_id, base_phi_deg in enumerate(base_azimuths_deg):
            for apd_group in range(4):
                # The sign is configurable because the manual provides
                # the 0.0181-degree magnitude but does not unambiguously
                # establish the sign in our coordinate convention.
                phi_deg = (
                    base_phi_deg
                    + apd_group_azimuth_sign
                    * apd_group
                    * apd_group_azimuth_offset_deg
                )

                for layer in range(4):
                    theta_deg = float(
                        self.scala2_polar_angle_deg(
                            azimuth_deg=phi_deg,
                            mirror_side=mirror_side,
                            apd_group=apd_group,
                            layer=layer,
                        )
                    )

                    phi_rad = np.deg2rad(phi_deg)
                    theta_rad = np.deg2rad(theta_deg)

                    # Sensor-local convention:
                    # x = forward, y = left, z = up.
                    direction_sensor = np.array(
                        [
                            np.cos(theta_rad) * np.cos(phi_rad),
                            np.cos(theta_rad) * np.sin(phi_rad),
                            np.sin(theta_rad),
                        ],
                        dtype=np.float64,
                    )

                    directions_sensor.append(direction_sensor)
                    ray_azimuths_deg.append(phi_deg)
                    ray_elevations_deg.append(theta_deg)
                    ray_apd_groups.append(apd_group)
                    ray_layers.append(layer)
                    ray_column_ids.append(column_id)

        directions_sensor = np.asarray(
            directions_sensor,
            dtype=np.float64,
        )

        # Each row is a direction vector, therefore use R.T on the right.
        directions_vehicle = (
            directions_sensor
            @ rotation_sensor_to_vehicle.T
        )

        # Numerical safety. Rotation should already preserve unit length.
        direction_norms = np.linalg.norm(
            directions_vehicle,
            axis=1,
            keepdims=True,
        )

        directions_vehicle = (
            directions_vehicle
            / np.maximum(direction_norms, 1e-12)
        )

        endpoints_vehicle = (
            start_point_xyz[None, :]
            + float(dist_m) * directions_vehicle
        )

        metadata = {
            "azimuth_deg_sensor": np.asarray(
                ray_azimuths_deg,
                dtype=np.float32,
            ),
            "elevation_deg_sensor": np.asarray(
                ray_elevations_deg,
                dtype=np.float32,
            ),
            "apd_group": np.asarray(
                ray_apd_groups,
                dtype=np.int8,
            ),
            "layer": np.asarray(
                ray_layers,
                dtype=np.int8,
            ),
            "column_id": np.asarray(
                ray_column_ids,
                dtype=np.int32,
            ),
            "mirror_side": np.full(
                len(directions_sensor),
                mirror_side,
                dtype=np.int8,
            ),
        }

        return endpoints_vehicle.astype(np.float32), metadata

    def build_scala_azimuths_deg(self, include_edges=False, include_center_edges=False):
        """
        Special for Scal2 Sensor.
        650 azimuths in total, 205 in each outer region, 240 in the center region.
        Returns azimuth angles (degrees) across the valid FOV in three regions:
        left  outer: [-66.5 or -66.25 ... -15.25]   step 0.25°
        center      : [-15.0 ... +15.0]             step 0.125° (±15 optional)
        right outer: [ +15.25 ... +66.25 or +66.5 ] step 0.25°
        Counts by option:
        650 = edges=False, center_edges=False
        651 = edges=False, center_edges=True
        652 = edges=True,  center_edges=False
        653 = edges=True,  center_edges=True
        """
        if include_edges:
            left  = np.arange(-66.5, -15.25 + 1e-9, 0.25)   # 206
            right = np.arange( 15.25,  66.5  + 1e-9, 0.25)  # 206
        else:
            left  = np.arange(-66.25, -15.25 + 1e-9, 0.25)  # 205
            right = np.arange( 15.25,  66.25 + 1e-9, 0.25)  # 205

        if include_center_edges:
            center = np.arange(-15.0, 15.0 + 1e-9, 0.125)   # 241 (includes +15.0)
        else:
            center = np.arange(-15.0, 15.0 - 0.125 + 1e-9, 0.125)  # 240

        az = np.concatenate([left, center, right]).astype(np.float32)
        assert np.all(np.diff(az) > 0)
        return az


    # def calculate_endpoints_vectorized_scala2(self, start_point_xyz, dist_m,
    #                                   horizontal_angles_deg, vertical_angles_deg,
    #                                   rotation_angle_deg):
        
    #     ha_vec = np.radians(horizontal_angles_deg + rotation_angle_deg)  # (H,)
    #     va_vec = np.radians(vertical_angles_deg)                          # (V,)

    #     hmesh, vmesh = np.meshgrid(ha_vec, va_vec, indexing='xy')  # both (V, H)

    #     dx = dist_m * np.cos(vmesh) * np.cos(hmesh)  # (V, H)
    #     dy = dist_m * np.cos(vmesh) * np.sin(hmesh)  # (V, H)
    #     dz = dist_m * np.sin(vmesh)                  # (V, H)

    #     endpoints = np.stack([dx, dy, dz], axis=-1).reshape(-1, 3)
    #     start_point_xyz = np.asarray(start_point_xyz, dtype=endpoints.dtype).reshape(1, 3)
    #     return endpoints + start_point_xyz

    def get_sensor_transforms(self, metadata, bev_padding=None):
        """
        Create homogeneous transformation matrices for base_link <-> sensor frames.

        Args:
            metadata (dict): {
                'start_point': (x, y, z),   # sensor position in base_link frame
                'rotation_angle': yaw_deg   # sensor yaw angle in degrees
            }

        Returns:
            T_baselink_from_sensor (np.ndarray): 4x4 matrix (base_link <- sensor)
            T_sensor_from_baselink (np.ndarray): 4x4 matrix (sensor <- base_link)
        """
        # Extract pose info
        sx, sy, sz = metadata['start_point']
        yaw_deg = metadata['rotation_angle']
        yaw_rad = math.radians(yaw_deg)

        # Build homogeneous transform (rotation + translation)
        T_baselink_from_sensor = tf_transformations.euler_matrix(0.0, 0.0, yaw_rad).astype(np.float32)

        if bev_padding is not None:
            T_baselink_from_sensor[0, 3] = np.float32(bev_padding - sx)
        else:
            T_baselink_from_sensor[0, 3] = np.float32(sx)

        T_baselink_from_sensor[1, 3] = np.float32(sy)
        T_baselink_from_sensor[2, 3] = np.float32(sz)

        # Inverse transform
        T_sensor_from_baselink = np.linalg.inv(T_baselink_from_sensor).astype(np.float32)

        return T_baselink_from_sensor, T_sensor_from_baselink
    
    def transform_pc_to_sensor_frame(self, pointcloud, metadata, bev_padding=None):

        _, T_sensor_from_baselink = self.get_sensor_transforms(metadata, bev_padding)

        pts = pointcloud[:, :3]
        N = pts.shape[0]

        ones = np.ones((N, 1), dtype=pts.dtype)
        pts_hom = np.hstack([pts, ones])  # (N,4)

        pts_sensor_hom = (T_sensor_from_baselink @ pts_hom.T).T  # (N,4)

        out = pointcloud.copy()          # (N,5): xyz + intensity + elongation (or other 2 features)
        out[:, :3] = pts_sensor_hom[:, :3]
        return out # 4x4 homogeneous transform: sensor in baselink

    def transform_plane_nd_to_sensor(self, metadata, bev_padding=None,
                                 normalize=True, enforce_up=True):
        """
        plane_baselink: (normal_b, d_b)
            normal_b: shape (3,), d_b: scalar
            plane equation in BASELINK: normal_b · x + d_b = 0

        returns: (normal_s, d_s) in SENSOR frame
            plane equation in SENSOR: normal_s · x + d_s = 0
        """

        #print(metadata['road_plane'])
        normal_b, d_b = metadata['road_plane']
        normal_b = np.asarray(normal_b, dtype=np.float64).reshape(3,)
        d_b = float(d_b)

        # Homogeneous plane vector π_b = [a,b,c,d]
        pi_b = np.array([normal_b[0], normal_b[1], normal_b[2], d_b], dtype=np.float64).reshape(4, 1)

        _, T_sensor_from_baselink = self.get_sensor_transforms(metadata, bev_padding)

        # Plane transform: π_s = T^{-T} π_b
        pi_s = (np.linalg.inv(T_sensor_from_baselink).T @ pi_b).reshape(4,)

        if normalize:
            n = pi_s[:3]
            nrm = np.linalg.norm(n)
            if nrm > 1e-12:
                pi_s = pi_s / nrm

        if enforce_up and pi_s[2] < 0:
            pi_s = -pi_s

        normal_s = pi_s[:3].astype(np.float32)
        d_s = float(pi_s[3])
        return normal_s, d_s

    def transform_point_cloud(
        self,
        sensor,
        original_pointcloud,
        start_point,
        dist,
        horizontal_angle_min,
        horizontal_angle_max,
        horizontal_rays=None,
        vertical_angles=None,
        rotation_angle=None,
        rotation_quaternion_xyzw=None,
        mirror_side=0,
        scala2_common=None,
        ray_debug_output_dir=None,
        ray_snapshot_interval=100,
        debug_apd_group=None,
        debug_layer=None,
    ):
        pointcloud_non_ground = original_pointcloud
        
        

        start_point = np.asarray(
            start_point,
            dtype=np.float32,
        )

        start_voxel = self.compute_voxel_coordinate(
            start_point,
            self.voxel_size,
        )

        voxel_points_map = self.create_voxel_map(
            pointcloud_non_ground
        )

        ray_metadata = None

        if sensor == "scala2":
            if rotation_quaternion_xyzw is None:
                raise ValueError(
                    "SCALA2 requires rotation_quaternion_xyzw"
                )

            if scala2_common is None:
                raise ValueError(
                    "SCALA2 requires scala2_common configuration"
                )

            all_endpoints_vec, ray_metadata = (
                self.calculate_endpoints_vectorized_scala2(
                    start_point_xyz=start_point,
                    dist_m=dist,
                    mirror_side=mirror_side,
                    rotation_quaternion_xyzw=(
                        rotation_quaternion_xyzw
                    ),
                    horizontal_angle_min=horizontal_angle_min,
                    horizontal_angle_max=horizontal_angle_max,
                    inner_angle_min=scala2_common.get(
                        "inner_angle_min",
                        -15.0,
                    ),
                    inner_angle_max=scala2_common.get(
                        "inner_angle_max",
                        15.0,
                    ),
                    outer_increment_deg=scala2_common.get(
                        "horizontal_increment_outer",
                        0.25,
                    ),
                    inner_increment_deg=scala2_common.get(
                        "horizontal_increment_inner",
                        0.125,
                    ),
                    apd_group_azimuth_offset_deg=(
                        scala2_common.get(
                            "apd_group_azimuth_offset",
                            0.0181,
                        )
                    ),
                    apd_group_azimuth_sign=(
                        scala2_common.get(
                            "apd_group_azimuth_sign",
                            1.0,
                        )
                    ),
                )
            )

        else:
            all_endpoints_vec = (
                self.calculate_endpoints_vectorized(
                    start_point=start_point,
                    dist=dist,
                    horizontal_angle_min=horizontal_angle_min,
                    horizontal_angle_max=horizontal_angle_max,
                    horizontal_rays=horizontal_rays,
                    vertical_angles=vertical_angles,
                )
            )

        # pointcloud_simulation = (
        #     self.simulate_point_cloud_with_original_point(
        #         start_point=start_point,
        #         all_endpoints=all_endpoints_vec,
        #         start_voxel=start_voxel,
        #         voxel_points_map=voxel_points_map,
        #     )
        # )
        
        # debug mode
        
        (
            pointcloud_simulation,
            debug_records,
            ray_hit_records,
        ) = self.simulate_point_cloud_with_original_point(
            start_point=start_point,
            all_endpoints=all_endpoints_vec,
            start_voxel=start_voxel,
            voxel_points_map=voxel_points_map,
            ray_metadata=ray_metadata,
            debug=True,
            debug_ray_indices=set(),
            snapshot_interval=ray_snapshot_interval,
            snapshot_directory=ray_debug_output_dir,
            debug_apd_group=debug_apd_group,
            debug_layer=debug_layer,
        )

        if ray_debug_output_dir is not None:
            os.makedirs(ray_debug_output_dir, exist_ok=True)
            self.save_ray_hit_records_csv(
                ray_hit_records=ray_hit_records,
                output_path=os.path.join(
                    ray_debug_output_dir,
                    "ray_hit_records.csv",
                ),
            )

        self.print_ray_debug_summary(
            debug_records,
            ground_z_threshold=0.0,
        )

        self.print_ground_hits_by_vertical_angle(
            debug_records,
            ground_z_threshold=0.0,
        )

       # Only for visualization and debugging purposes
        # print("=" * 60)
        # print("Point Cloud Simulation")
        # print("=" * 60)

        # print(f"Type: {type(pointcloud_simulation)}")
        # print(f"Shape: {pointcloud_simulation.shape}")
        # print(f"Dtype: {pointcloud_simulation.dtype}")
        # print(f"Number of points: {len(pointcloud_simulation)}")


        # # Create a unique identifier for this function call.
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        # file_identifier = (
        #     f"{sensor}"
        #     f"_mirror_{mirror_side}"
        #     f"_{timestamp}"
        # )


        # base_save_dir = (
        #     "/home/samanti/git_repos/object_detection_dl/"
        #     "quasi2d-lidar-simulation/test_data_output/"
        #     "visualizing_outputs"
        # )

        # save_dir_original = os.path.join(
        #     base_save_dir,
        #     "original",
        # )

        # save_dir_transformed = os.path.join(
        #     base_save_dir,
        #     "transformed",
        # )

        # os.makedirs(save_dir_original, exist_ok=True)
        # os.makedirs(save_dir_transformed, exist_ok=True)


        # original_save_path = os.path.join(
        #     save_dir_original,
        #     f"original_{file_identifier}.npy",
        # )

        # transformed_save_path = os.path.join(
        #     save_dir_transformed,
        #     f"pointcloud_simulation_{file_identifier}.npy",
        # )


        # np.save(
        #     original_save_path,
        #     np.asarray(pointcloud_non_ground),
        # )

        # np.save(
        #     transformed_save_path,
        #     np.asarray(pointcloud_simulation),
        # )


        # print(f"Saved original point cloud to: {original_save_path}")
        # print(f"Saved simulated point cloud to: {transformed_save_path}")
        
        return np.asarray(
            pointcloud_simulation,
            dtype=np.float32,
        ), ray_metadata


    def point_on_ray_at_same_distance(self, A, B, P):
        """
        Given points A, B (defining a ray starting at A), and an arbitrary point P,
        return Q on the ray AB such that distance(A, Q) == distance(A, P).

        Parameters
        ----------
        A, B, P : array-like of shape (3,)
            3D coordinates of the start of the ray A, another point on the ray B,
            and the target point P, respectively.

        Returns
        -------
        Q : ndarray of shape (3,)
            The point on the ray AB at the same distance from A as P is.
        
        Raises
        ------
        ValueError
            If A and B coincide (zero-length ray).
        """
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)
        P = np.asarray(P, dtype=float)

        # Distance from A to P
        d = np.linalg.norm(P - A)
        #print(f"Distance from A to P: {d}")

        # Unit direction from A toward B
        AB = B - A
        L = np.linalg.norm(AB)
        if L == 0:
            raise ValueError("Start point A and direction point B must be distinct.")
        u = AB / L

        # Move from A along the ray by distance d
        return A + d * u


    def _pick_representative_point_on_ray(self, start_point_xyz, end_point_xyz, points_in_voxel_xyz):
        """
        Choose the point in this voxel that best matches the ray:
        1) minimal perpendicular distance to the ray
        2) tie-break: minimal along-ray distance (closest 'first-return')
        Returns (idx, perp_dist, t_along) where point ~= start + t*dir, t>=0.
        """
        s = np.asarray(start_point_xyz, dtype=np.float32)
        e = np.asarray(end_point_xyz, dtype=np.float32)
        P = np.asarray(points_in_voxel_xyz, dtype=np.float32)  # (N,3)

        d = e - s
        dd = float(np.dot(d, d))
        if dd < 1e-12 or P.shape[0] == 0:
            return None, None, None

        # projection parameter t for each point onto the ray direction (not clamped yet)
        # t = ((p-s)·d) / (d·d)
        t = ((P - s) @ d) / dd

        # consider only points in front of the sensor along the ray
        valid = t >= 0.0
        if not np.any(valid):
            return None, None, None

        P_v = P[valid]
        t_v = t[valid]

        # closest point on the ray line for each candidate
        Q = s + t_v[:, None] * d[None, :]
        perp = np.linalg.norm(P_v - Q, axis=1)

        # rank by perp distance, then by along-ray t (smaller = earlier hit)
        order = np.lexsort((t_v, perp))
        best_local = order[0]

        # map back to original indices
        valid_indices = np.nonzero(valid)[0]
        best_idx = int(valid_indices[best_local])
        return best_idx, float(perp[best_local]), float(t[best_idx])


    def simulate_point_cloud_with_original_point(
        self,
        start_point,
        all_endpoints,
        start_voxel,
        voxel_points_map,
        ray_metadata=None,
        debug=False,
        debug_ray_indices=None,
        max_candidate_points_to_store=20,
        snapshot_interval=None,
        snapshot_directory=None,
        debug_apd_group=None,
        debug_layer=None,
    ):
        """
        Simulate one LiDAR return per emitted ray.

        For every ray, the first occupied traversed voxel that produces a valid
        representative point is treated as the ray hit.

        Parameters
        ----------
        start_point : array-like
            LiDAR origin in the point-cloud coordinate frame.

        all_endpoints : array-like, shape (N, 3)
            Maximum-range endpoint for every emitted ray.

        start_voxel : tuple
            Voxel coordinate containing the LiDAR origin.

        voxel_points_map : dict
            Mapping:
                voxel coordinate -> original points inside that voxel.

        ray_metadata : optional
            Metadata corresponding to each ray. Ideally contains horizontal and
            vertical angles and any SCALA2-specific identifiers.

        debug : bool
            Whether to collect detailed ray-level debugging information.

        debug_ray_indices : set[int] or None
            Rays for which detailed candidate information should be stored.
            None means all rays.

        max_candidate_points_to_store : int
            Limits stored candidate points per occupied voxel.
        """

        pointcloud_simulation = []
        pointcloud_simulation_set = set()
        debug_records = []
        
        
        # Compact record of the point selected by each ray.
        ray_hit_records = []

        start_xyz = np.asarray(
            start_point,
            dtype=np.float32,
        )

        all_endpoints = np.asarray(
            all_endpoints,
            dtype=np.float32,
        )

        for ray_idx, end_xyz in enumerate(all_endpoints):
            end_xyz = np.asarray(
                end_xyz,
                dtype=np.float32,
            )

            metadata = (
                self.get_scala2_ray_metadata(
                    ray_metadata=ray_metadata,
                    ray_idx=ray_idx,
                )
                if ray_metadata is not None
                else None
            )

            # Optional single-channel debugging. With both values set, only
            # rays from that SCALA2 (APD group, layer) are simulated.
            if metadata is not None:
                if (
                    debug_apd_group is not None
                    and metadata["apd_group"] != debug_apd_group
                ):
                    continue
                if (
                    debug_layer is not None
                    and metadata["layer"] != debug_layer
                ):
                    continue

            should_debug_ray = (
                debug
                and (
                    debug_ray_indices is None
                    or ray_idx in debug_ray_indices
                )
            )

            ray_vector = end_xyz - start_xyz
            ray_length = float(np.linalg.norm(ray_vector))

            if ray_length <= 1e-8:
                if debug:
                    debug_records.append({
                        "ray_index": ray_idx,
                        "metadata": metadata,
                        "origin": start_xyz.copy(),
                        "endpoint": end_xyz.copy(),
                        "status": "invalid_zero_length_ray",
                        "selected_point": None,
                    })
                ray_hit_records.append({
                    "ray_index": int(ray_idx),
                    "status": "invalid_zero_length_ray",
                    "column_id": None if metadata is None else metadata["column_id"],
                    "apd_group": None if metadata is None else metadata["apd_group"],
                    "layer": None if metadata is None else metadata["layer"],
                    "mirror_side": None if metadata is None else metadata["mirror_side"],
                    "azimuth_deg_sensor": None if metadata is None else metadata["azimuth_deg_sensor"],
                    "elevation_deg_sensor": None if metadata is None else metadata["elevation_deg_sensor"],
                })
                continue

            ray_direction = ray_vector / ray_length
            
            
            # Vehicle-frame elevation after quaternion rotation
            
            actual_elevation_deg_vehicle = float(
                np.degrees(
                    np.arctan2(
                        ray_direction[2],
                        np.linalg.norm(ray_direction[:2]),
                    )
                )
            )

            end_voxel = self.compute_voxel_coordinate(
                end_xyz,
                self.voxel_size,
            )

            intersected_voxels = self.bresenham3D(
                start_voxel,
                end_voxel,
            )

            ray_record = None

            if debug:
                ray_record = {
                    "ray_index": ray_idx,
                    "metadata": metadata,
                    "origin": start_xyz.copy(),
                    "endpoint": end_xyz.copy(),
                    "ray_direction": ray_direction.copy(),
                    "ray_length": ray_length,
                    "actual_elevation_deg_vehicle": actual_elevation_deg_vehicle,
                    "start_voxel": tuple(start_voxel),
                    "end_voxel": tuple(end_voxel),
                    "number_of_traversed_voxels": len(
                        intersected_voxels
                    ),
                    "traversed_voxels": (
                        [tuple(v) for v in intersected_voxels]
                        if should_debug_ray
                        else None
                    ),
                    "occupied_voxels": [],
                    "selected_voxel": None,
                    "representative_original_point": None,
                    "projected_point": None,
                    "representative_distance_from_sensor": None,
                    "projected_distance_from_sensor": None,
                    "representative_perpendicular_distance": None,
                    "projection_displacement": None,
                    "status": "no_hit",
                }

            hit_found = False

            for voxel_step, voxel_coord in enumerate(
                intersected_voxels
            ):
                voxel_coord = tuple(voxel_coord)

                if voxel_coord not in voxel_points_map:
                    continue

                points_in_voxel = np.asarray(
                    voxel_points_map[voxel_coord]
                )

                occupied_voxel_record = None

                if should_debug_ray:
                    occupied_voxel_record = {
                        "voxel_step": voxel_step,
                        "voxel_coordinate": voxel_coord,
                        "raw_shape": tuple(points_in_voxel.shape),
                        "status": None,
                        "candidate_points": None,
                        "selected_candidate_index": None,
                    }

                if (
                    points_in_voxel.ndim != 2
                    or points_in_voxel.shape[0] == 0
                ):
                    if should_debug_ray:
                        occupied_voxel_record["status"] = (
                            "invalid_or_empty"
                        )
                        ray_record["occupied_voxels"].append(
                            occupied_voxel_record
                        )
                    continue

                if points_in_voxel.shape[1] < 3:
                    if should_debug_ray:
                        occupied_voxel_record["status"] = (
                            "fewer_than_three_coordinates"
                        )
                        ray_record["occupied_voxels"].append(
                            occupied_voxel_record
                        )
                    continue

                P_xyz = points_in_voxel[:, :3]

                (
                    best_idx,
                    best_perpendicular_distance,
                    best_t_along_ray,
                ) = self._pick_representative_point_on_ray(
                    start_xyz,
                    end_xyz,
                    P_xyz,
                )

                if should_debug_ray:
                    candidate_debug = (
                        self._build_candidate_ray_debug(
                            start_xyz=start_xyz,
                            ray_direction=ray_direction,
                            candidate_points=P_xyz,
                        )
                    )

                    occupied_voxel_record["candidate_points"] = (
                        candidate_debug[
                            :max_candidate_points_to_store
                        ]
                    )

                    occupied_voxel_record[
                        "selected_candidate_index"
                    ] = (
                        None
                        if best_idx is None
                        else int(best_idx)
                    )

                    occupied_voxel_record[
                        "best_perpendicular_distance"
                    ] = (
                        None
                        if best_perpendicular_distance is None
                        else float(best_perpendicular_distance)
                    )

                    occupied_voxel_record[
                        "best_t_along_ray"
                    ] = (
                        None
                        if best_t_along_ray is None
                        else float(best_t_along_ray)
                    )


                if best_idx is None:
                    if should_debug_ray:
                        occupied_voxel_record["status"] = (
                            "no_valid_representative"
                        )
                        ray_record["occupied_voxels"].append(
                            occupied_voxel_record
                        )
                    continue

                rep = points_in_voxel[best_idx]
                rep_xyz = np.asarray(
                    rep[:3],
                    dtype=np.float32,
                )

                projected_xyz = (
                    self.point_on_ray_at_same_distance(
                        start_xyz,
                        end_xyz,
                        rep_xyz,
                    )
                )

                projected_xyz = np.asarray(
                    projected_xyz,
                    dtype=np.float32,
                )

                if points_in_voxel.shape[1] >= 4:
                    intensity = float(rep[3])
                else:
                    intensity = 0.0

                out = np.array(
                    [
                        projected_xyz[0],
                        projected_xyz[1],
                        projected_xyz[2],
                        intensity,
                    ],
                    dtype=np.float32,
                )

                key = (
                    float(out[0]),
                    float(out[1]),
                    float(out[2]),
                )

                rep_relative = rep_xyz - start_xyz

                rep_projection_distance = float(
                    np.dot(
                        rep_relative,
                        ray_direction,
                    )
                )

                closest_point_on_ray = (
                    start_xyz
                    + rep_projection_distance * ray_direction
                )

                rep_perpendicular_distance = float(
                    np.linalg.norm(
                        rep_xyz - closest_point_on_ray
                    )
                )

                rep_distance = float(
                    np.linalg.norm(
                        rep_xyz - start_xyz
                    )
                )

                projected_distance = float(
                    np.linalg.norm(
                        projected_xyz - start_xyz
                    )
                )

                projection_displacement = float(
                    np.linalg.norm(
                        projected_xyz - rep_xyz
                    )
                )

                rep_elevation_deg = float(
                    np.degrees(
                        np.arctan2(
                            rep_relative[2],
                            np.linalg.norm(rep_relative[:2]),
                        )
                    )
                )

                ray_hit_records.append({
                    "ray_index": int(ray_idx),
                    "status": "hit",
                    "column_id": None if metadata is None else metadata["column_id"],
                    "apd_group": None if metadata is None else metadata["apd_group"],
                    "layer": None if metadata is None else metadata["layer"],
                    "channel_id": (
                        None
                        if metadata is None
                        else metadata["apd_group"] * 4 + metadata["layer"]
                    ),
                    "mirror_side": None if metadata is None else metadata["mirror_side"],
                    "azimuth_deg_sensor": None if metadata is None else metadata["azimuth_deg_sensor"],
                    "elevation_deg_sensor": None if metadata is None else metadata["elevation_deg_sensor"],
                    "elevation_deg_vehicle": actual_elevation_deg_vehicle,
                    "selected_voxel_x": int(voxel_coord[0]),
                    "selected_voxel_y": int(voxel_coord[1]),
                    "selected_voxel_z": int(voxel_coord[2]),
                    "representative_x": float(rep_xyz[0]),
                    "representative_y": float(rep_xyz[1]),
                    "representative_z": float(rep_xyz[2]),
                    "projected_x": float(projected_xyz[0]),
                    "projected_y": float(projected_xyz[1]),
                    "projected_z": float(projected_xyz[2]),
                    "representative_range_m": rep_distance,
                    "representative_elevation_deg": rep_elevation_deg,
                    "perpendicular_distance_m": rep_perpendicular_distance,
                    "projection_displacement_m": projection_displacement,
                })

                if should_debug_ray:
                    occupied_voxel_record["status"] = "selected"
                    ray_record["occupied_voxels"].append(
                        occupied_voxel_record
                    )

                if key not in pointcloud_simulation_set:
                    pointcloud_simulation.append(out)
                    pointcloud_simulation_set.add(key)

                    if debug:
                        ray_record["status"] = "hit_added"
                else:
                    if debug:
                        ray_record["status"] = "hit_duplicate"

                if debug:
                    ray_record["selected_voxel"] = voxel_coord
                    ray_record["selected_voxel_step"] = voxel_step

                    ray_record[
                        "representative_original_point"
                    ] = rep.copy()

                    ray_record["projected_point"] = out.copy()

                    ray_record[
                        "representative_distance_from_sensor"
                    ] = rep_distance

                    ray_record[
                        "projected_distance_from_sensor"
                    ] = projected_distance

                    ray_record[
                        "representative_perpendicular_distance"
                    ] = rep_perpendicular_distance

                    ray_record[
                        "projection_displacement"
                    ] = projection_displacement

                hit_found = True

                # A ray must generate at most one hit.
                break

            if not hit_found:
                ray_hit_records.append({
                    "ray_index": int(ray_idx),
                    "status": "no_hit",
                    "column_id": None if metadata is None else metadata["column_id"],
                    "apd_group": None if metadata is None else metadata["apd_group"],
                    "layer": None if metadata is None else metadata["layer"],
                    "channel_id": (
                        None
                        if metadata is None
                        else metadata["apd_group"] * 4 + metadata["layer"]
                    ),
                    "mirror_side": None if metadata is None else metadata["mirror_side"],
                    "azimuth_deg_sensor": None if metadata is None else metadata["azimuth_deg_sensor"],
                    "elevation_deg_sensor": None if metadata is None else metadata["elevation_deg_sensor"],
                    "elevation_deg_vehicle": actual_elevation_deg_vehicle,
                })

            if debug:
                if not hit_found:
                    ray_record["status"] = "no_hit"

                debug_records.append(ray_record)

            if (
                snapshot_directory is not None
                and snapshot_interval is not None
                and snapshot_interval > 0
                and (
                    (ray_idx + 1) % snapshot_interval == 0
                    or ray_idx == len(all_endpoints) - 1
                )
            ):
                os.makedirs(snapshot_directory, exist_ok=True)
                snapshot = np.asarray(
                    pointcloud_simulation,
                    dtype=np.float32,
                )
                np.save(
                    os.path.join(
                        snapshot_directory,
                        f"rays_000000_to_{ray_idx:06d}.npy",
                    ),
                    snapshot,
                )

        pointcloud_simulation = np.asarray(
            pointcloud_simulation,
            dtype=np.float32,
        )

        if debug:
            return (
                pointcloud_simulation,
                debug_records,
                ray_hit_records,
            )

        return pointcloud_simulation
    
    
    def save_ray_hit_records_csv(
        self,
        ray_hit_records,
        output_path,
    ):
        """Save the ray-to-hit mapping in a human-readable CSV file."""
        if not ray_hit_records:
            print("No ray-hit records to save.")
            return

        output_directory = os.path.dirname(output_path)
        if output_directory:
            os.makedirs(output_directory, exist_ok=True)

        fieldnames = []
        for record in ray_hit_records:
            for key in record:
                if key not in fieldnames:
                    fieldnames.append(key)

        with open(output_path, "w", newline="") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=fieldnames,
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(ray_hit_records)

        print(
            f"Saved {len(ray_hit_records)} ray records to: "
            f"{output_path}"
        )

    def _build_candidate_ray_debug(
        self,
        start_xyz,
        ray_direction,
        candidate_points,
    ):
        candidate_debug = []

        start_xyz = np.asarray(
            start_xyz,
            dtype=np.float32,
        )

        ray_direction = np.asarray(
            ray_direction,
            dtype=np.float32,
        )

        ray_direction_norm = np.linalg.norm(
            ray_direction
        )

        if ray_direction_norm <= 1e-8:
            return candidate_debug

        ray_direction = (
            ray_direction / ray_direction_norm
        )

        for candidate_idx, point_xyz in enumerate(
            candidate_points
        ):
            point_xyz = np.asarray(
                point_xyz,
                dtype=np.float32,
            )

            relative = point_xyz - start_xyz

            projection_distance = float(
                np.dot(
                    relative,
                    ray_direction,
                )
            )

            closest_point = (
                start_xyz
                + projection_distance * ray_direction
            )

            perpendicular_distance = float(
                np.linalg.norm(
                    point_xyz - closest_point
                )
            )

            sensor_distance = float(
                np.linalg.norm(relative)
            )

            candidate_debug.append({
                "candidate_index": candidate_idx,
                "point_xyz": point_xyz.copy(),
                "projection_distance": projection_distance,
                "perpendicular_distance": (
                    perpendicular_distance
                ),
                "sensor_distance": sensor_distance,
                "in_front_of_sensor": (
                    projection_distance > 0.0
                ),
            })

        return candidate_debug
    
    def print_ray_debug_summary(
        self,
        debug_records,
        ground_z_threshold=0.0,
    ):
        total_rays = len(debug_records)

        hit_records = [
            record
            for record in debug_records
            if record["status"] in {
                "hit_added",
                "hit_duplicate",
            }
        ]

        no_hit_records = [
            record
            for record in debug_records
            if record["status"] == "no_hit"
        ]

        ground_records = [
            record
            for record in hit_records
            if record["projected_point"] is not None
            and record["projected_point"][2]
            <= ground_z_threshold
        ]

        print("\n========== RAY DEBUG SUMMARY ==========")
        print(f"Total rays: {total_rays}")
        print(f"Rays with hits: {len(hit_records)}")
        print(f"Rays without hits: {len(no_hit_records)}")
        print(f"Ground-like hits: {len(ground_records)}")

        perpendicular_distances = [
            record[
                "representative_perpendicular_distance"
            ]
            for record in hit_records
            if record[
                "representative_perpendicular_distance"
            ] is not None
        ]

        projection_displacements = [
            record["projection_displacement"]
            for record in hit_records
            if record["projection_displacement"] is not None
        ]

        if perpendicular_distances:
            print(
                "Representative point-to-ray distance:"
            )
            print(
                f"  minimum: "
                f"{np.min(perpendicular_distances):.4f} m"
            )
            print(
                f"  mean: "
                f"{np.mean(perpendicular_distances):.4f} m"
            )
            print(
                f"  maximum: "
                f"{np.max(perpendicular_distances):.4f} m"
            )

        if projection_displacements:
            print("Original-to-projected displacement:")
            print(
                f"  minimum: "
                f"{np.min(projection_displacements):.4f} m"
            )
            print(
                f"  mean: "
                f"{np.mean(projection_displacements):.4f} m"
            )
            print(
                f"  maximum: "
                f"{np.max(projection_displacements):.4f} m"
            )

        print("=======================================\n")

    def print_ground_hits_by_vertical_angle(
        self,
        debug_records,
        ground_z_threshold=0.0,
    ):
        channel_groups = {}

        for record in debug_records:
            projected_point = record.get("projected_point")

            if projected_point is None:
                continue

            projected_point = np.asarray(
                projected_point,
                dtype=np.float32,
            )

            # The fourth value is intensity, so inspect z at index 2.
            if projected_point[2] > ground_z_threshold:
                continue

            metadata = record.get("metadata")

            if metadata is None:
                continue

            sensor_elevation = float(
                metadata["elevation_deg_sensor"]
            )

            vehicle_elevation = float(
                record["actual_elevation_deg_vehicle"]
            )

            apd_group = int(metadata["apd_group"])
            layer = int(metadata["layer"])
            mirror_side = int(metadata["mirror_side"])

            channel_key = (
                mirror_side,
                apd_group,
                layer,
            )

            horizontal_range = float(
                np.linalg.norm(
                    projected_point[:2]
                    - record["origin"][:2]
                )
            )

            if channel_key not in channel_groups:
                channel_groups[channel_key] = {
                    "count": 0,
                    "sensor_elevations": [],
                    "vehicle_elevations": [],
                    "horizontal_ranges": [],
                    "ray_indices": [],
                }

            group = channel_groups[channel_key]

            group["count"] += 1
            group["sensor_elevations"].append(
                sensor_elevation
            )
            group["vehicle_elevations"].append(
                vehicle_elevation
            )
            group["horizontal_ranges"].append(
                horizontal_range
            )
            group["ray_indices"].append(
                record["ray_index"]
            )

        print(
            "\n===== GROUND HITS BY SCALA2 CHANNEL ====="
        )

        if not channel_groups:
            print(
                "No simulated points were below the "
                f"ground threshold z={ground_z_threshold:.3f} m."
            )

        for channel_key in sorted(channel_groups):
            mirror_side, apd_group, layer = channel_key
            group = channel_groups[channel_key]

            sensor_elevations = np.asarray(
                group["sensor_elevations"]
            )

            vehicle_elevations = np.asarray(
                group["vehicle_elevations"]
            )

            horizontal_ranges = np.asarray(
                group["horizontal_ranges"]
            )

            print(
                f"\nMirror={mirror_side}, "
                f"APD={apd_group}, "
                f"layer={layer}"
            )

            print(
                f"  Number of ground hits: "
                f"{group['count']}"
            )

            print(
                "  Sensor elevation: "
                f"{sensor_elevations.min():+.4f} to "
                f"{sensor_elevations.max():+.4f} deg"
            )

            print(
                "  Vehicle elevation: "
                f"{vehicle_elevations.min():+.4f} to "
                f"{vehicle_elevations.max():+.4f} deg"
            )

            print(
                "  Horizontal range: "
                f"{horizontal_ranges.min():.3f} to "
                f"{horizontal_ranges.max():.3f} m, "
                f"mean={horizontal_ranges.mean():.3f} m"
            )

        print(
            "\n========================================\n"
        )
    
    def bresenham3D(self, start, end):
        """
        Bresenham's line algorithm in 3D.
        Yields all voxel coordinates on the line from start to end.
        """
        x1, y1, z1 = start
        x2, y2, z2 = end
        points = []
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        dz = abs(z2 - z1)
        xs = 1 if x2 > x1 else -1
        ys = 1 if y2 > y1 else -1
        zs = 1 if z2 > z1 else -1

        # X-axis
        if dx >= dy and dx >= dz:
            p1 = 2*dy - dx
            p2 = 2*dz - dx
            while x1 != x2:
                x1 += xs
                if p1 >= 0:
                    y1 += ys
                    p1 -= 2*dx
                if p2 >= 0:
                    z1 += zs
                    p2 -= 2*dx
                p1 += 2*dy
                p2 += 2*dz
                points.append((x1, y1, z1))

        # Y-axis
        elif dy >= dx and dy >= dz:
            p1 = 2*dx - dy
            p2 = 2*dz - dy
            while y1 != y2:
                y1 += ys
                if p1 >= 0:
                    x1 += xs
                    p1 -= 2*dy
                if p2 >= 0:
                    z1 += zs
                    p2 -= 2*dy
                p1 += 2*dx
                p2 += 2*dz
                points.append((x1, y1, z1))

        # Z-axis
        else:
            p1 = 2*dy - dz
            p2 = 2*dx - dz
            while z1 != z2:
                z1 += zs
                if p1 >= 0:
                    y1 += ys
                    p1 -= 2*dz
                if p2 >= 0:
                    x1 += xs
                    p2 -= 2*dz
                p1 += 2*dy
                p2 += 2*dx
                points.append((x1, y1, z1))

        return points

    def compute_voxel_coordinate(self, point, voxel_size):
        return tuple(np.floor(point / voxel_size).astype(int))


    def calculate_endpoint(self, x0, y0, z0, distance, horiz_angle, vert_angle):
        # angles to radians
        horiz_angle_rad = np.radians(horiz_angle)
        vert_angle_rad = np.radians(vert_angle)
        
        # calculate changes based on angles
        dx = distance * np.cos(vert_angle_rad) * np.cos(horiz_angle_rad)
        dy = distance * np.cos(vert_angle_rad) * np.sin(horiz_angle_rad)
        dz = distance * np.sin(vert_angle_rad)

        #print(dx,dy,dz)
        
        # endpoint coordinates
        x_end = x0 + dx
        y_end = y0 + dy
        z_end = z0 + dz
        
        return x_end, y_end, z_end
    

    def create_voxel_map(self, pointcloud_non_ground):
        points_xyz = pointcloud_non_ground[:, 0:3]
        voxel_coords = self.compute_voxel_coordinate(points_xyz, self.voxel_size)

        voxel_points_map = {}
        for voxel_coord, point in zip(voxel_coords, pointcloud_non_ground):
            voxel_coord_tuple = tuple(voxel_coord)
            if voxel_coord_tuple in voxel_points_map:
                voxel_points_map[voxel_coord_tuple].append(point)
            else:
                voxel_points_map[voxel_coord_tuple] = [point]

        return voxel_points_map

    
    def calculate_endpoints_vectorized(self, start_point, dist, horizontal_angle_min, horizontal_angle_max, horizontal_rays, vertical_angles):
        horizontal_angles = np.linspace(horizontal_angle_min, horizontal_angle_max, num=horizontal_rays+1)
        horizontal_angles_rounded = np.round(horizontal_angles, 2)
        horizontal_angles_radians = np.radians(horizontal_angles_rounded)
        vertical_angles_radians = np.radians(vertical_angles)

        start_point = np.array(start_point)[None, None, :]

        horiz_mesh, vert_mesh = np.meshgrid(horizontal_angles_radians, vertical_angles_radians)
        dx = (dist * np.cos(vert_mesh) * np.cos(horiz_mesh)).flatten()
        dy = (dist * np.cos(vert_mesh) * np.sin(horiz_mesh)).flatten()
        dz = (dist * np.sin(vert_mesh)).flatten()
        endpoints = np.stack([dx, dy, dz], axis=-1) + start_point

        return endpoints.reshape(-1, 3)

    def find_nearest_point_to_center(self, points_in_voxel, voxel_coord, voxel_size):
        # find the nearest point to the voxel center
        voxel_center = (np.asarray(voxel_coord) + 0.5) * voxel_size
        distances = [np.linalg.norm(point - voxel_center) for point in points_in_voxel]
        nearest_point_index = np.argmin(distances)
        return points_in_voxel[nearest_point_index]

    def find_mean_of_points_in_voxel(self, points_in_voxel):

        # calculate the mean of all points in the voxel
        points_in_voxel_np = np.array([point[:3] for point in points_in_voxel])  # extract 3D coordinates (x, y, z)
        mean_point = np.mean(points_in_voxel_np, axis=0)  # mean across all points in the voxel

        return mean_point

    def filter_bounding_boxes(
        self,
        frame,
        current_pointcloud,
        sensor_metadata,
        horizontal_fov_min,
        horizontal_fov_max,
        vertical_fov_min=None,
        vertical_fov_max=None,
        threshold=4,
    ):
        """
        Filter Waymo boxes using the full sensor pose and the simulated
        point cloud.

        Boxes are first checked against the sensor-local angular FoV,
        then retained only if they contain at least `threshold` points
        from the simulated cloud.
        """
        filtered_bounding_boxes = []

        for label in frame.laser_labels:
            if label.type not in (1, 2, 4):
                continue

            box_data = {
                "center": np.array(
                    [
                        label.box.center_x,
                        label.box.center_y,
                        label.box.center_z,
                    ],
                    dtype=np.float64,
                ),
                "dimensions": np.array(
                    [
                        label.box.length,
                        label.box.width,
                        label.box.height,
                    ],
                    dtype=np.float64,
                ),
                "orientation": (
                    tf_transformations.quaternion_from_euler(
                        0.0,
                        0.0,
                        label.box.heading,
                    )
                ),
                "heading": label.box.heading,
                "velocity": [
                    label.metadata.speed_x,
                    label.metadata.speed_y,
                    label.metadata.speed_z,
                ],
                "acceleration": [
                    label.metadata.accel_x,
                    label.metadata.accel_y,
                    label.metadata.accel_z,
                ],
                "type": label.type,
                "color": self.get_label_color(label.type),
                "track_id": label.id,
                "num_lidar_points_in_box": (
                    label.num_lidar_points_in_box
                ),
                "difficulty": (
                    label.detection_difficulty_level
                ),
                "tracking_difficulty": (
                    label.tracking_difficulty_level
                ),
            }

            if not self.is_bounding_box_in_sensor_fov(
                box_data=box_data,
                sensor_metadata=sensor_metadata,
                horizontal_fov_min=horizontal_fov_min,
                horizontal_fov_max=horizontal_fov_max,
                vertical_fov_min=vertical_fov_min,
                vertical_fov_max=vertical_fov_max,
            ):
                continue

            points_in_box = self.box_containing_points(
                box_data,
                current_pointcloud,
            )

            if points_in_box >= threshold:
                box_data[
                    "num_lidar_points_in_box_filtered"
                ] = points_in_box

                filtered_bounding_boxes.append(box_data)

        return filtered_bounding_boxes
    
    def box_containing_points(self, box_data, point_cloud):
        count = 0
        for point in point_cloud:
            if self.is_point_in_box(point, box_data):
                count += 1
        return count
    
    def is_point_in_box(self, point, box_data):
        # convert the point to the box coordinate frame
        point_rel = point[:3] - box_data['center']
        point_rot = self.rotate_vector(point_rel, box_data['orientation'])
        # check if the point is inside the box
        half_size = box_data['dimensions'] / 2
        return all(-half_size[i] <= point_rot[i] <= half_size[i] for i in range(3))

    def is_bounding_box_in_sensor_fov(
        self,
        box_data,
        sensor_metadata,
        horizontal_fov_min,
        horizontal_fov_max,
        vertical_fov_min=None,
        vertical_fov_max=None,
    ):
        """
        Check whether the box center lies inside the sensor-local FoV.

        The box center is originally expressed in the Waymo base-link
        frame. It is transformed into the sensor frame using the full
        translation and quaternion.
        """
        center_baselink = np.asarray(
            box_data["center"],
            dtype=np.float64,
        ).reshape(3)

        sensor_position = np.asarray(
            sensor_metadata["start_point"],
            dtype=np.float64,
        ).reshape(3)

        quaternion_xyzw = sensor_metadata.get(
            "rotation_quaternion_xyzw"
        )

        if quaternion_xyzw is not None:
            qx, qy, qz, qw = quaternion_xyzw

            R_baselink_from_sensor = (
                self.quaternion_xyzw_to_rotation_matrix(
                    qx=qx,
                    qy=qy,
                    qz=qz,
                    qw=qw,
                )
            )
        else:
            # Legacy yaw-only sensor support.
            yaw_rad = np.deg2rad(
                sensor_metadata["rotation_angle"]
            )

            R_baselink_from_sensor = (
                tf_transformations.euler_matrix(
                    0.0,
                    0.0,
                    yaw_rad,
                )[:3, :3]
            )

        # Column-vector form:
        # p_sensor = R^T (p_baselink - t)
        center_sensor = (
            R_baselink_from_sensor.T
            @ (center_baselink - sensor_position)
        )

        x_sensor, y_sensor, z_sensor = center_sensor

        # Reject boxes behind the sensor.
        if x_sensor <= 0.0:
            return False

        azimuth_deg = np.rad2deg(
            np.arctan2(y_sensor, x_sensor)
        )

        horizontal_distance = np.hypot(
            x_sensor,
            y_sensor,
        )

        elevation_deg = np.rad2deg(
            np.arctan2(
                z_sensor,
                horizontal_distance,
            )
        )

        in_horizontal_fov = (
            horizontal_fov_min
            <= azimuth_deg
            <= horizontal_fov_max
        )

        if not in_horizontal_fov:
            return False

        if (
            vertical_fov_min is not None
            and elevation_deg < vertical_fov_min
        ):
            return False

        if (
            vertical_fov_max is not None
            and elevation_deg > vertical_fov_max
        ):
            return False

        return True
    
    def rotate_vector(self, vector, quaternion):
        # rotate the vector by the given quaternion
        rotated_vector = tf_transformations.quaternion_multiply(
            tf_transformations.quaternion_multiply(quaternion, np.append(vector, 0)),
            tf_transformations.quaternion_conjugate(quaternion)
        )
        return rotated_vector[:3]


    def get_label_color(self, label_type):
        if label_type == 1: # vehicle, green
            return [0.0, 1.0, 0.0] 
        elif label_type == 2: # pedestrian, blue
            return [0.0, 0.0, 1.0]
        elif label_type == 4: # cyclist
            return [0.0, 0.5, 0.5]
        
        
    def get_scala2_ray_metadata(
        self,
        ray_metadata,
        ray_idx,
    ):
        if ray_metadata is None:
            return None

        return {
            "azimuth_deg_sensor": float(
                ray_metadata["azimuth_deg_sensor"][ray_idx]
            ),
            "elevation_deg_sensor": float(
                ray_metadata["elevation_deg_sensor"][ray_idx]
            ),
            "apd_group": int(
                ray_metadata["apd_group"][ray_idx]
            ),
            "layer": int(
                ray_metadata["layer"][ray_idx]
            ),
            "column_id": int(
                ray_metadata["column_id"][ray_idx]
            ),
            "mirror_side": int(
                ray_metadata["mirror_side"][ray_idx]
            ),
        }



