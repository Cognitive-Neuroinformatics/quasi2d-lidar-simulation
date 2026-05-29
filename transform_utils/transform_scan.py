import numpy as np
import cv2 as cv
import transformations as tf_transformations
from transform_utils.math_utils import transform, rot2, get_position_and_yaw, pose_boxminus
from utils import gt_sampling_utils

class TransformScan3D():
    def __init__(self, sensor):
        self.sensor = sensor

    def relative_ego_motion(self, prev_pose_4x4: np.ndarray,
                        curr_pose_4x4: np.ndarray,
                        pose_convention: str = "vehicle_to_world") -> np.ndarray:
        """
        Compute relative transform that maps points from previous vehicle(baselink) frame -> current vehicle(baselink) frame.

        pose_convention:
        - "vehicle_to_world": pose maps vehicle frame -> world frame  (Waymo frame.pose.transform is typically this)
                T_curr_from_prev = inv(T_w_from_curr) @ T_w_from_prev
        - "world_to_vehicle": pose maps world frame -> vehicle frame
                T_curr_from_prev = T_curr_from_w @ inv(T_prev_from_w)
        """
        prev_pose_4x4 = np.asarray(prev_pose_4x4, dtype=np.float64).reshape(4, 4)
        curr_pose_4x4 = np.asarray(curr_pose_4x4, dtype=np.float64).reshape(4, 4)

        if pose_convention == "vehicle_to_world":
            T_curr_from_prev = np.linalg.inv(curr_pose_4x4) @ prev_pose_4x4
        elif pose_convention == "world_to_vehicle":
            T_curr_from_prev = curr_pose_4x4 @ np.linalg.inv(prev_pose_4x4)
        else:
            raise ValueError(f"Unknown pose_convention: {pose_convention}")

        return T_curr_from_prev


    def transform_points_4x4(self, points: np.ndarray, T_4x4: np.ndarray) -> np.ndarray:
        """
        Apply 4x4 transform to a point cloud.

        Accepts:
        - points shape (N,3) or (N,>=4). Only xyz are transformed.
            Extra channels (e.g., intensity, elongation, time) are preserved unchanged.

        Returns transformed points with same shape as input.
        """
        if points is None:
            return None
        points = np.asarray(points)
        if points.size == 0:
            return points.copy()

        T_4x4 = np.asarray(T_4x4, dtype=np.float64).reshape(4, 4)

        xyz = points[:, :3].astype(np.float64)
        ones = np.ones((xyz.shape[0], 1), dtype=np.float64)
        xyz1 = np.hstack([xyz, ones])                      # (N,4)
        xyz1_t = (T_4x4 @ xyz1.T).T                        # (N,4)
        xyz_t = xyz1_t[:, :3].astype(points.dtype)

        out = points.copy()
        out[:, :3] = xyz_t
        return out


    def transform_scan(self, pose, scan):
        transformed_scan = np.zeros((scan.shape[0], 3))
        for i in range(scan.shape[0]):
            transformed_scan[i,:] = transform(pose, scan[i, :3])
        return transformed_scan



class TransformScan2D():
    def __init__(self, sensor, x_size, y_size, z_min, z_max, resolution):
        self.sensor = sensor
        self.x_size = x_size
        self.y_size = y_size
        self.z_min = z_min
        self.z_max = z_max
        self.resolution = resolution


    def transform_scan(self, pose, scan):
        transformed_scan = np.zeros((scan.shape[0], 3))
        for i in range(scan.shape[0]):
            transformed_scan[i, :2] = transform(pose, scan[i, :2])  # only x, y transformation
            transformed_scan[i, 2] = scan[i, 2]  # retain the z value for height coloring in combined method
        return transformed_scan


    def sensor_conjugate_SE2(self, vehicle_diff_pose, T_baselink_from_sensor):
        """
        Convert vehicle-frame relative motion to sensor-frame relative motion.
        Inputs:
        vehicle_diff_pose: [tx_v, ty_v, yaw] (current<-prev) in current baselink or global (?) frame
        T_baselink_from_sensor: 4x4 (vehicle <- sensor), i.e. sensor->baselink extrinsic
        Returns:
        diff_sensor_xy_yaw: [tx_s, ty_s, yaw] to apply directly on sensor points
        """
        tx_v, ty_v, yaw = vehicle_diff_pose
        R_rel = rot2(yaw)
        t_v   = np.array([tx_v, ty_v])

        # sensor->vehicle extrinsic
        R_vs = T_baselink_from_sensor[:2,:2]     # 2x2
        r    = T_baselink_from_sensor[:2, 3]     # (sx, sy) in vehicle coords

        # Conjugation in 2D:
        # yaw is unchanged; translation gets the offset correction
        I2 = np.eye(2)
        t_s = R_vs.T @ (t_v + (R_rel - I2) @ r)

        return np.array([t_s[0], t_s[1], yaw])

    def point2pixel(self, point):
            """ Convert 3D point to 2D pixel coordinates based on sensor type """
            if self.sensor in ['scala', 't7']:
                pixel_x = int(point[0] / self.resolution)
                pixel_y = int((self.y_size / 2 - point[1] / self.resolution))
                return [pixel_x, pixel_y]

            elif self.sensor in ['vlp16', 'vlp32']:
                x_mid = self.x_size / 2
                y_mid = self.y_size / 2
                pixel_x = int(x_mid + point[0] / self.resolution)
                pixel_y = int(y_mid - point[1] / self.resolution)
                return [pixel_x, pixel_y]      

    def calculate_corners(self, center, dimensions, quaternion):
        # quaternion to rotation matrix
        rotation_matrix = tf_transformations.quaternion_matrix(quaternion)[:3, :3]
        length, width = dimensions[0], dimensions[1]
        # calculate corner points in local frame
        half_length = length / 2.0
        half_width = width / 2.0
        corners_local = np.array([
            [half_length, half_width],
            [-half_length, half_width],
            [-half_length, -half_width],
            [half_length, -half_width]
        ])
        # apply rotation
        corners_global = np.dot(corners_local, rotation_matrix[:2, :2].T)
        # translate to global position
        corners_global[:, 0] += center[0]
        corners_global[:, 1] += center[1]
        
        return corners_global.flatten()
    
    
    
    def transform_odom(
        self,
        scan,
        pose,
        method,
        prev_pose=None,
        prev_pointcloud=None,
        T_baselink_from_sensor=None
    ):
        pose = get_position_and_yaw(pose)

        img_blue = np.zeros((int(self.x_size), int(self.y_size)), np.uint8)
        img_green = np.zeros((int(self.x_size), int(self.y_size)), np.uint8)
        img_red = np.zeros((int(self.x_size), int(self.y_size)), np.uint8)

        if method == "height":
            for point in scan:
                x_s, y_s = self.point2pixel(point[:2])
                px_s, py_s = int(x_s), int(y_s)

                z_value = point[2]
                normalized_z = (z_value - self.z_min) / (self.z_max - self.z_min)
                normalized_z = np.clip(normalized_z, 0, 1)

                blue_value = int(255 * (1 - normalized_z))
                red_value = int(255 * normalized_z)

                cv.circle(img_blue, (px_s, py_s), 1, blue_value, -1)
                cv.circle(img_red, (px_s, py_s), 1, red_value, -1)

            img = cv.merge((img_blue, img_green, img_red))

        elif method == "overlap":
            if prev_pose is not None and prev_pointcloud is not None:
                prev_pose = get_position_and_yaw(prev_pose)
                diff_pose = pose_boxminus(prev_pose, pose)

                if T_baselink_from_sensor is not None:
                    diff_s = self.sensor_conjugate_SE2(
                        diff_pose,
                        T_baselink_from_sensor
                    )
                    transformed_prev_scan_with_swing = self.transform_scan(
                        diff_s,
                        prev_pointcloud
                    )
                else:
                    transformed_prev_scan_with_swing = self.transform_scan(
                        diff_pose,
                        prev_pointcloud
                    )

                for point in scan:
                    x_s, y_s = self.point2pixel(point[:2])
                    px_s, py_s = int(x_s), int(y_s)
                    cv.circle(img_green, (px_s, py_s), 1, 255, -1)

                for point in transformed_prev_scan_with_swing:
                    x_s, y_s = self.point2pixel(point[:2])
                    px_s, py_s = int(x_s), int(y_s)
                    cv.circle(img_red, (px_s, py_s), 1, 255, -1)

            else:
                for point in scan:
                    x_s, y_s = self.point2pixel(point[:2])
                    px_s, py_s = int(x_s), int(y_s)
                    cv.circle(img_green, (px_s, py_s), 1, 255, -1)

            img = cv.merge((img_blue, img_green, img_red))

        elif method == "overlap_height":
            if prev_pose is not None and prev_pointcloud is not None:
                prev_pose = get_position_and_yaw(prev_pose)
                diff_pose = pose_boxminus(prev_pose, pose)

                if T_baselink_from_sensor is not None:
                    diff_s = self.sensor_conjugate_SE2(
                        diff_pose,
                        T_baselink_from_sensor
                    )
                    transformed_prev_scan_with_swing = self.transform_scan(
                        diff_s,
                        prev_pointcloud
                    )
                else:
                    transformed_prev_scan_with_swing = self.transform_scan(
                        diff_pose,
                        prev_pointcloud
                    )

                # current frame: green channel, intensity varies with z
                for point in scan:
                    x_s, y_s = self.point2pixel(point[:2])
                    px_s, py_s = int(x_s), int(y_s)

                    z_value = point[2]
                    normalized_z = (z_value - self.z_min) / (self.z_max - self.z_min)
                    normalized_z = np.clip(normalized_z, 0, 1)

                    green_intensity = int(255 * normalized_z)

                    cv.circle(img_green, (px_s, py_s), 1, green_intensity, -1)

                # previous transformed frame: red channel, intensity varies with z
                for point in transformed_prev_scan_with_swing:
                    x_s, y_s = self.point2pixel(point[:2])
                    px_s, py_s = int(x_s), int(y_s)

                    z_value = point[2]
                    normalized_z = (z_value - self.z_min) / (self.z_max - self.z_min)
                    normalized_z = np.clip(normalized_z, 0, 1)

                    red_intensity = int(255 * normalized_z)

                    cv.circle(img_red, (px_s, py_s), 1, red_intensity, -1)

            else:
                # first frame: only current scan, green intensity varies with z
                for point in scan:
                    x_s, y_s = self.point2pixel(point[:2])
                    px_s, py_s = int(x_s), int(y_s)

                    z_value = point[2]
                    normalized_z = (z_value - self.z_min) / (self.z_max - self.z_min)
                    normalized_z = np.clip(normalized_z, 0, 1)

                    green_intensity = int(255 * normalized_z)

                    cv.circle(img_green, (px_s, py_s), 1, green_intensity, -1)

            img = cv.merge((img_blue, img_green, img_red))

        else:
            raise ValueError(f"Unknown transform_odom method: {method}")

        return img