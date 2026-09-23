import math
import numpy as np
import transformations as tf_transformations


class PointCloudTransformer:
    """
    Point-cloud simulator for the legacy SCALA sensor and SCALA Gen-2.

    SCALA2 conventions
    ------------------
    Sensor-local frame:
        x = forward
        y = left
        z = up

    A SCALA2 scan is generated as:
        base azimuth positions
            x 4 APD groups (0.0181 deg stagger by default)
            x 4 layers per APD group

    The polar/elevation angle is computed from the SCALA Gen-2 manual
    equation for every individual ray and the full sensor quaternion is
    used to rotate rays into the base_link frame.
    """

    def __init__(self, voxel_size, ground_removal_method="default"):
        self.voxel_size = voxel_size
        self.ground_removal_method = ground_removal_method
        self.current_transformed_pointcloud = None

    # ------------------------------------------------------------------
    # SCALA2 scan-pattern generation
    # ------------------------------------------------------------------

    @staticmethod
    def scala2_polar_angle_deg(
        azimuth_deg,
        mirror_side,
        apd_group,
        layer,
    ):
        """SCALA Gen-2 polar/elevation angle (manual Equation 1)."""
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

        channel_term = (
            0.6025 * layer
            + 2.564 * apd_group
            - 4.749
        )

        mirror_sign = 1.0 if mirror_side == 0 else -1.0
        return channel_term + mirror_sign * mirror_term

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
        Generate the non-uniform SCALA2 nominal azimuth grid.

        With the default endpoint convention this gives 653 nominal positions:
          206 left outer + 240 center + 207 right outer.
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

        return np.unique(np.round(azimuths, decimals=8))

    @staticmethod
    def quaternion_xyzw_to_rotation_matrix(qx, qy, qz, qw):
        """Return the sensor->base_link 3x3 rotation matrix."""
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
        Generate SCALA2 ray endpoints in base_link.

        For every base azimuth:
            4 APD-group azimuths are produced,
            each APD group contains 4 vertical layers.
        """
        start_point_xyz = np.asarray(
            start_point_xyz,
            dtype=np.float64,
        ).reshape(3)

        qx, qy, qz, qw = rotation_quaternion_xyzw
        R_baselink_from_sensor = self.quaternion_xyzw_to_rotation_matrix(
            qx, qy, qz, qw
        )

        base_azimuths_deg = self.generate_scala2_base_azimuths_deg(
            horizontal_angle_min=horizontal_angle_min,
            horizontal_angle_max=horizontal_angle_max,
            inner_angle_min=inner_angle_min,
            inner_angle_max=inner_angle_max,
            outer_increment=outer_increment_deg,
            inner_increment=inner_increment_deg,
        )

        num_base = len(base_azimuths_deg)
        num_rays = num_base * 4 * 4

        directions_sensor = np.empty((num_rays, 3), dtype=np.float64)
        ray_azimuths_deg = np.empty(num_rays, dtype=np.float32)
        ray_elevations_deg = np.empty(num_rays, dtype=np.float32)
        ray_apd_groups = np.empty(num_rays, dtype=np.int8)
        ray_layers = np.empty(num_rays, dtype=np.int8)
        ray_column_ids = np.empty(num_rays, dtype=np.int32)

        idx = 0
        for column_id, base_phi_deg in enumerate(base_azimuths_deg):
            for apd_group in range(4):
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

                    directions_sensor[idx] = (
                        np.cos(theta_rad) * np.cos(phi_rad),
                        np.cos(theta_rad) * np.sin(phi_rad),
                        np.sin(theta_rad),
                    )

                    ray_azimuths_deg[idx] = phi_deg
                    ray_elevations_deg[idx] = theta_deg
                    ray_apd_groups[idx] = apd_group
                    ray_layers[idx] = layer
                    ray_column_ids[idx] = column_id
                    idx += 1

        directions_baselink = directions_sensor @ R_baselink_from_sensor.T
        directions_baselink /= np.maximum(
            np.linalg.norm(directions_baselink, axis=1, keepdims=True),
            1e-12,
        )

        endpoints_baselink = (
            start_point_xyz[None, :]
            + float(dist_m) * directions_baselink
        )

        metadata = {
            "azimuth_deg_sensor": ray_azimuths_deg,
            "elevation_deg_sensor": ray_elevations_deg,
            "apd_group": ray_apd_groups,
            "layer": ray_layers,
            "column_id": ray_column_ids,
            "mirror_side": np.full(
                num_rays,
                mirror_side,
                dtype=np.int8,
            ),
        }

        return endpoints_baselink.astype(np.float32), metadata

    # ------------------------------------------------------------------
    # Sensor transforms
    # ------------------------------------------------------------------

    def get_sensor_transforms(self, metadata, bev_padding=None):
        """
        Build sensor <-> base_link homogeneous transforms.

        SCALA2 uses the full quaternion. Legacy SCALA falls back to yaw-only.
        """
        sx, sy, sz = np.asarray(
            metadata["start_point"],
            dtype=np.float64,
        )

        quaternion_xyzw = metadata.get("rotation_quaternion_xyzw")

        if quaternion_xyzw is not None:
            qx, qy, qz, qw = quaternion_xyzw
            R_baselink_from_sensor = self.quaternion_xyzw_to_rotation_matrix(
                qx, qy, qz, qw
            )
        else:
            yaw_rad = np.deg2rad(float(metadata["rotation_angle"]))
            R_baselink_from_sensor = tf_transformations.euler_matrix(
                0.0, 0.0, yaw_rad
            )[:3, :3]

        T_baselink_from_sensor = np.eye(4, dtype=np.float64)
        T_baselink_from_sensor[:3, :3] = R_baselink_from_sensor

        # Retained for compatibility with the existing BEV/label pipeline.
        if bev_padding is not None:
            T_baselink_from_sensor[0, 3] = float(bev_padding) - sx
        else:
            T_baselink_from_sensor[0, 3] = sx

        T_baselink_from_sensor[1, 3] = sy
        T_baselink_from_sensor[2, 3] = sz

        T_sensor_from_baselink = np.linalg.inv(T_baselink_from_sensor)

        return (
            T_baselink_from_sensor.astype(np.float32),
            T_sensor_from_baselink.astype(np.float32),
        )

    def transform_pc_to_sensor_frame(
        self,
        pointcloud,
        metadata,
        bev_padding=None,
    ):
        _, T_sensor_from_baselink = self.get_sensor_transforms(
            metadata,
            bev_padding,
        )

        pts = pointcloud[:, :3]
        ones = np.ones((pts.shape[0], 1), dtype=pts.dtype)
        pts_hom = np.hstack([pts, ones])

        pts_sensor_hom = (
            T_sensor_from_baselink @ pts_hom.T
        ).T

        out = pointcloud.copy()
        out[:, :3] = pts_sensor_hom[:, :3]
        return out

    def transform_plane_nd_to_sensor(
        self,
        metadata,
        bev_padding=None,
        normalize=True,
        enforce_up=True,
    ):
        normal_b, d_b = metadata["road_plane"]
        normal_b = np.asarray(normal_b, dtype=np.float64).reshape(3)
        d_b = float(d_b)

        pi_b = np.array(
            [normal_b[0], normal_b[1], normal_b[2], d_b],
            dtype=np.float64,
        ).reshape(4, 1)

        _, T_sensor_from_baselink = self.get_sensor_transforms(
            metadata,
            bev_padding,
        )

        pi_s = (
            np.linalg.inv(T_sensor_from_baselink).T @ pi_b
        ).reshape(4)

        if normalize:
            norm = np.linalg.norm(pi_s[:3])
            if norm > 1e-12:
                pi_s /= norm

        if enforce_up and pi_s[2] < 0:
            pi_s = -pi_s

        return pi_s[:3].astype(np.float32), float(pi_s[3])

    # ------------------------------------------------------------------
    # Point-cloud simulation
    # ------------------------------------------------------------------

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
        projection_method="same_distance",
    ):
        start_point = np.asarray(start_point, dtype=np.float32)

        start_voxel = self.compute_voxel_coordinate(
            start_point,
            self.voxel_size,
        )
        voxel_points_map = self.create_voxel_map(original_pointcloud)

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

            horizontal_increment_cfg = scala2_common.get(
                "horizontal_increment",
                {},
            )

            outer_increment = scala2_common.get(
                "horizontal_increment_outer",
                horizontal_increment_cfg.get("outer", 0.25),
            )
            inner_increment = scala2_common.get(
                "horizontal_increment_inner",
                horizontal_increment_cfg.get("inner", 0.125),
            )

            all_endpoints_vec, ray_metadata = (
                self.calculate_endpoints_vectorized_scala2(
                    start_point_xyz=start_point,
                    dist_m=dist,
                    mirror_side=mirror_side,
                    rotation_quaternion_xyzw=rotation_quaternion_xyzw,
                    horizontal_angle_min=horizontal_angle_min,
                    horizontal_angle_max=horizontal_angle_max,
                    inner_angle_min=scala2_common.get(
                        "inner_angle_min", -15.0
                    ),
                    inner_angle_max=scala2_common.get(
                        "inner_angle_max", 15.0
                    ),
                    outer_increment_deg=float(outer_increment),
                    inner_increment_deg=float(inner_increment),
                    apd_group_azimuth_offset_deg=float(
                        scala2_common.get(
                            "apd_group_azimuth_offset",
                            scala2_common.get(
                                "apd_group_azimuth_offset_deg",
                                0.0181,
                            ),
                        )
                    ),
                    apd_group_azimuth_sign=float(
                        scala2_common.get(
                            "apd_group_azimuth_sign",
                            1.0,
                        )
                    ),
                )
            )
        else:
            all_endpoints_vec = self.calculate_endpoints_vectorized(
                start_point=start_point,
                dist=dist,
                horizontal_angle_min=horizontal_angle_min,
                horizontal_angle_max=horizontal_angle_max,
                horizontal_rays=horizontal_rays,
                vertical_angles=vertical_angles,
            )

        pointcloud_simulation = (
            self.simulate_point_cloud_with_original_point(
                start_point=start_point,
                all_endpoints=all_endpoints_vec,
                start_voxel=start_voxel,
                voxel_points_map=voxel_points_map,
                projection_method=projection_method,
            )
        )

        return (
            np.asarray(pointcloud_simulation, dtype=np.float32),
            ray_metadata,
        )

    def _pick_representative_point_on_ray(
        self,
        start_point_xyz,
        end_point_xyz,
        points_in_voxel_xyz,
    ):
        """
        Pick the point nearest to the ray in the current occupied voxel.
        Ties are resolved in favor of the earlier point along the ray.
        """
        s = np.asarray(start_point_xyz, dtype=np.float32)
        e = np.asarray(end_point_xyz, dtype=np.float32)
        P = np.asarray(points_in_voxel_xyz, dtype=np.float32)

        d = e - s
        dd = float(np.dot(d, d))

        if dd < 1e-12 or P.shape[0] == 0:
            return None

        t = ((P - s) @ d) / dd
        valid = t >= 0.0

        if not np.any(valid):
            return None

        valid_indices = np.nonzero(valid)[0]
        P_valid = P[valid]
        t_valid = t[valid]

        closest = s + t_valid[:, None] * d[None, :]
        perpendicular = np.linalg.norm(
            P_valid - closest,
            axis=1,
        )

        order = np.lexsort((t_valid, perpendicular))
        return int(valid_indices[order[0]])

    def simulate_point_cloud_with_original_point(
        self,
        start_point,
        all_endpoints,
        start_voxel,
        voxel_points_map,
        projection_method="same_distance",
    ):
        """Cast one ray per endpoint and keep at most one first return."""
        pointcloud_simulation = []
        pointcloud_simulation_set = set()

        start_xyz = np.asarray(start_point, dtype=np.float32)

        for end_xyz in np.asarray(all_endpoints, dtype=np.float32):
            end_voxel = self.compute_voxel_coordinate(
                end_xyz,
                self.voxel_size,
            )

            for voxel_coord in self.bresenham3D(
                start_voxel,
                end_voxel,
            ):
                if voxel_coord not in voxel_points_map:
                    continue

                points_in_voxel = np.asarray(
                    voxel_points_map[voxel_coord]
                )

                if (
                    points_in_voxel.ndim != 2
                    or points_in_voxel.shape[0] == 0
                    or points_in_voxel.shape[1] < 3
                ):
                    continue

                best_idx = self._pick_representative_point_on_ray(
                    start_xyz,
                    end_xyz,
                    points_in_voxel[:, :3],
                )

                if best_idx is None:
                    continue

                rep = points_in_voxel[best_idx]
                
                if projection_method == "same_distance":

                    projected_xyz = self.point_on_ray_at_same_distance(
                        start_xyz,
                        end_xyz,
                        rep[:3],
                    )

                elif projection_method == "orthogonal":

                    projected_xyz = self.point_on_ray_orthogonal_projection(
                        start_xyz,
                        end_xyz,
                        rep[:3],
                    )

                    if projected_xyz is None:
                        continue

                else:
                    raise ValueError(
                        f"Unknown projection_method: {projection_method}. "
                        "Choose 'same_distance' or 'orthogonal'."
                    )

                intensity = (
                    float(rep[3])
                    if points_in_voxel.shape[1] >= 4
                    else 0.0
                )

                out = np.array(
                    [
                        projected_xyz[0],
                        projected_xyz[1],
                        projected_xyz[2],
                        intensity,
                    ],
                    dtype=np.float32,
                )

                key = tuple(float(v) for v in out[:3])

                if key not in pointcloud_simulation_set:
                    pointcloud_simulation.append(out)
                    pointcloud_simulation_set.add(key)

                break

        return np.asarray(pointcloud_simulation, dtype=np.float32)

    @staticmethod
    def point_on_ray_at_same_distance(A, B, P):
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)
        P = np.asarray(P, dtype=float)

        distance = np.linalg.norm(P - A)
        ray = B - A
        length = np.linalg.norm(ray)

        if length == 0:
            raise ValueError(
                "Start point and endpoint must be distinct."
            )

        return A + distance * (ray / length)
    
    @staticmethod
    def point_on_ray_orthogonal_projection(A, B, P):
        """
        Orthogonally project point P onto ray AB.

        The returned point lies exactly on the ray and uses the
        along-ray projection distance.
        """
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)
        P = np.asarray(P, dtype=float)

        ray = B - A
        ray_length = np.linalg.norm(ray)

        if ray_length < 1e-12:
            raise ValueError(
                "Start point and endpoint must be distinct."
            )

        ray_unit = ray / ray_length

        along_ray_distance = np.dot(
            P - A,
            ray_unit,
        )

        if along_ray_distance < 0.0:
            return None

        return A + along_ray_distance * ray_unit

    @staticmethod
    def compute_voxel_coordinate(point, voxel_size):
        return tuple(
            np.floor(point / voxel_size).astype(int)
        )

    def create_voxel_map(self, pointcloud):
        voxel_coords = self.compute_voxel_coordinate(
            pointcloud[:, :3],
            self.voxel_size,
        )

        voxel_points_map = {}

        for voxel_coord, point in zip(
            voxel_coords,
            pointcloud,
        ):
            key = tuple(voxel_coord)
            voxel_points_map.setdefault(key, []).append(point)

        return voxel_points_map

    @staticmethod
    def calculate_endpoints_vectorized(
        start_point,
        dist,
        horizontal_angle_min,
        horizontal_angle_max,
        horizontal_rays,
        vertical_angles,
    ):
        """Legacy uniform-grid SCALA endpoint generation."""
        horizontal_angles = np.linspace(
            horizontal_angle_min,
            horizontal_angle_max,
            num=horizontal_rays + 1,
        )

        horizontal_angles = np.radians(
            np.round(horizontal_angles, 2)
        )
        vertical_angles = np.radians(vertical_angles)

        hmesh, vmesh = np.meshgrid(
            horizontal_angles,
            vertical_angles,
        )

        dx = dist * np.cos(vmesh) * np.cos(hmesh)
        dy = dist * np.cos(vmesh) * np.sin(hmesh)
        dz = dist * np.sin(vmesh)

        directions = np.stack(
            [dx, dy, dz],
            axis=-1,
        ).reshape(-1, 3)

        return directions + np.asarray(
            start_point,
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Bounding-box filtering used by the processor
    # ------------------------------------------------------------------

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
                "orientation": tf_transformations.quaternion_from_euler(
                    0.0,
                    0.0,
                    label.box.heading,
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
                "difficulty": label.detection_difficulty_level,
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

    def is_bounding_box_in_sensor_fov(
        self,
        box_data,
        sensor_metadata,
        horizontal_fov_min,
        horizontal_fov_max,
        vertical_fov_min=None,
        vertical_fov_max=None,
    ):
        center_baselink = np.asarray(
            box_data["center"],
            dtype=np.float64,
        )
        sensor_position = np.asarray(
            sensor_metadata["start_point"],
            dtype=np.float64,
        )

        quaternion_xyzw = sensor_metadata.get(
            "rotation_quaternion_xyzw"
        )

        if quaternion_xyzw is not None:
            R_baselink_from_sensor = (
                self.quaternion_xyzw_to_rotation_matrix(
                    *quaternion_xyzw
                )
            )
        else:
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

        center_sensor = (
            R_baselink_from_sensor.T
            @ (center_baselink - sensor_position)
        )

        x_sensor, y_sensor, z_sensor = center_sensor

        if x_sensor <= 0.0:
            return False

        azimuth_deg = np.rad2deg(
            np.arctan2(y_sensor, x_sensor)
        )
        elevation_deg = np.rad2deg(
            np.arctan2(
                z_sensor,
                np.hypot(x_sensor, y_sensor),
            )
        )

        if not (
            horizontal_fov_min
            <= azimuth_deg
            <= horizontal_fov_max
        ):
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

    def box_containing_points(self, box_data, point_cloud):
        return sum(
            self.is_point_in_box(point, box_data)
            for point in point_cloud
        )

    def is_point_in_box(self, point, box_data):
        point_rel = point[:3] - box_data["center"]
        point_rot = self.rotate_vector(
            point_rel,
            box_data["orientation"],
        )
        half_size = box_data["dimensions"] / 2.0

        return all(
            -half_size[i] <= point_rot[i] <= half_size[i]
            for i in range(3)
        )

    @staticmethod
    def rotate_vector(vector, quaternion):
        rotated = tf_transformations.quaternion_multiply(
            tf_transformations.quaternion_multiply(
                quaternion,
                np.append(vector, 0),
            ),
            tf_transformations.quaternion_conjugate(
                quaternion
            ),
        )
        return rotated[:3]

    @staticmethod
    def get_label_color(label_type):
        if label_type == 1:
            return [0.0, 1.0, 0.0]
        if label_type == 2:
            return [0.0, 0.0, 1.0]
        if label_type == 4:
            return [0.0, 0.5, 0.5]
        return [1.0, 1.0, 1.0]

    # ------------------------------------------------------------------
    # Bresenham
    # ------------------------------------------------------------------

    def bresenham3D(self, start, end):
        x1, y1, z1 = start
        x2, y2, z2 = end
        points = []

        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        dz = abs(z2 - z1)

        xs = 1 if x2 > x1 else -1
        ys = 1 if y2 > y1 else -1
        zs = 1 if z2 > z1 else -1

        if dx >= dy and dx >= dz:
            p1 = 2 * dy - dx
            p2 = 2 * dz - dx

            while x1 != x2:
                x1 += xs

                if p1 >= 0:
                    y1 += ys
                    p1 -= 2 * dx

                if p2 >= 0:
                    z1 += zs
                    p2 -= 2 * dx

                p1 += 2 * dy
                p2 += 2 * dz
                points.append((x1, y1, z1))

        elif dy >= dx and dy >= dz:
            p1 = 2 * dx - dy
            p2 = 2 * dz - dy

            while y1 != y2:
                y1 += ys

                if p1 >= 0:
                    x1 += xs
                    p1 -= 2 * dy

                if p2 >= 0:
                    z1 += zs
                    p2 -= 2 * dy

                p1 += 2 * dx
                p2 += 2 * dz
                points.append((x1, y1, z1))

        else:
            p1 = 2 * dy - dz
            p2 = 2 * dx - dz

            while z1 != z2:
                z1 += zs

                if p1 >= 0:
                    y1 += ys
                    p1 -= 2 * dz

                if p2 >= 0:
                    x1 += xs
                    p2 -= 2 * dz

                p1 += 2 * dy
                p2 += 2 * dx
                points.append((x1, y1, z1))

        return points