import os
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


class PointCloudTransformer:
    def __init__(self, voxel_size, ground_removal_method='default'):
        self.voxel_size = voxel_size
        self.ground_removal_method = ground_removal_method
        self.current_transformed_pointcloud = None
        
        # Just for Scala2 sensor - building it once and caching it
        self.az_deg = self.build_scala_azimuths_deg(include_edges=True, include_center_edges=True)  


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


    def calculate_endpoints_vectorized_scala2(self, start_point_xyz, dist_m,
                                      horizontal_angles_deg, vertical_angles_deg,
                                      rotation_angle_deg):
        
        ha_vec = np.radians(horizontal_angles_deg + rotation_angle_deg)  # (H,)
        va_vec = np.radians(vertical_angles_deg)                          # (V,)

        hmesh, vmesh = np.meshgrid(ha_vec, va_vec, indexing='xy')  # both (V, H)

        dx = dist_m * np.cos(vmesh) * np.cos(hmesh)  # (V, H)
        dy = dist_m * np.cos(vmesh) * np.sin(hmesh)  # (V, H)
        dz = dist_m * np.sin(vmesh)                  # (V, H)

        endpoints = np.stack([dx, dy, dz], axis=-1).reshape(-1, 3)
        start_point_xyz = np.asarray(start_point_xyz, dtype=endpoints.dtype).reshape(1, 3)
        return endpoints + start_point_xyz

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

    def transform_point_cloud(self, sensor, original_pointcloud, start_point, dist, horizontal_angle_min, horizontal_angle_max, horizontal_rays, vertical_angles, rotation_angle):

        pointcloud_non_ground = original_pointcloud # ground removal functionality paused
        start_voxel = self.compute_voxel_coordinate(np.asarray(start_point), self.voxel_size)
        voxel_points_map = self.create_voxel_map(pointcloud_non_ground)

        if sensor == 'scala2': # Non Uniform azimuth grid
            
            all_endpoints_vec = self.calculate_endpoints_vectorized_scala2(
                start_point_xyz=start_point,
                dist_m=dist,
                horizontal_angles_deg=self.az_deg,
                vertical_angles_deg=np.asarray(vertical_angles, dtype=np.float32),
                rotation_angle_deg=rotation_angle  # yaw of the sensor
            )
        else:
            # Legacy: Uniform azimuth grid
            all_endpoints_vec = self.calculate_endpoints_vectorized(start_point, dist, horizontal_angle_min, horizontal_angle_max, horizontal_rays, vertical_angles)
        
        pointcloud_simulation = self.simulate_point_cloud_with_original_point(start_point,all_endpoints_vec, start_voxel, voxel_points_map)
        
        pointcloud_simulation = np.array(pointcloud_simulation)
        
        return pointcloud_simulation


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


    def simulate_point_cloud_with_original_point(self, start_point, all_endpoints, start_voxel, voxel_points_map):
        
        pointcloud_simulation = []
        pointcloud_simulation_set = set()

        start_xyz = np.asarray(start_point, dtype=np.float32)

        for end_xyz in all_endpoints:
            end_xyz = np.asarray(end_xyz, dtype=np.float32)

            end_voxel = self.compute_voxel_coordinate(end_xyz, self.voxel_size)
            intersected_voxels = self.bresenham3D(start_voxel, end_voxel)

            for voxel_coord in intersected_voxels:
                if voxel_coord not in voxel_points_map:
                    continue

                points_in_voxel = np.asarray(voxel_points_map[voxel_coord])

                if points_in_voxel.ndim != 2 or points_in_voxel.shape[0] == 0:
                    continue

                if points_in_voxel.shape[1] >= 3:
                    P_xyz = points_in_voxel[:, :3]
                else:
                    continue

                best_idx, _, _ = self._pick_representative_point_on_ray(start_xyz, end_xyz, P_xyz)
                if best_idx is None:
                    continue

                rep = points_in_voxel[best_idx]

                rep_xyz = rep[:3]
                projected_xyz = self.point_on_ray_at_same_distance(start_xyz, end_xyz, rep_xyz)

                if points_in_voxel.shape[1] >= 4:
                    intensity = float(rep[3])
                else:
                    intensity = 0.0
 

                # just taking intensity info and not elongation as not available in our data
                out = np.array([projected_xyz[0], projected_xyz[1], projected_xyz[2],
                                intensity], dtype=np.float32)

                key = (float(out[0]), float(out[1]), float(out[2]))
                if key not in pointcloud_simulation_set:
                    pointcloud_simulation.append(out)
                    pointcloud_simulation_set.add(key)

                break  

        return np.asarray(pointcloud_simulation, dtype=np.float32)
    
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

    def filter_bounding_boxes(self, frame, current_pointcloud, sensor_position, sensor_rotation_angle, horizontal_fov_min, horizontal_fov_max, threshold):
        """Filter bounding boxes based on the transformed pointcloud FOV.
        
        Args:
            frame (waymo_open_dataset.dataset_pb2.Frame): current waymo frame
            pointcloud (np.ndarray): transformed point cloud to filter bounding boxes
            threshold (int): threshold of how many points should be in a bbox
        """
        filtered_bounding_boxes = []
        for i, label in enumerate(frame.laser_labels):
            if label.type in [1, 2, 4]:  # Filter für relevante Typen
                box_data = {
                    'center': np.array([label.box.center_x, label.box.center_y, label.box.center_z]),
                    'dimensions': np.array([label.box.length, label.box.width, label.box.height]),
                    'orientation': tf_transformations.quaternion_from_euler(0, 0, label.box.heading),
                    'heading': label.box.heading,
                    'velocity': [label.metadata.speed_x, label.metadata.speed_y, label.metadata.speed_z],  
                    'acceleration': [label.metadata.accel_x, label.metadata.accel_y, label.metadata.accel_z], 
                    'type': label.type,
                    'color': self.get_label_color(label.type),
                    'track_id': label.id,
                    'num_lidar_points_in_box': label.num_lidar_points_in_box,  # original points
                    'difficulty': label.detection_difficulty_level,
                    'tracking_difficulty': label.tracking_difficulty_level,
                }

                # additional check if the bounding box is within the sensors FOV
                if not self.is_bounding_box_in_sensor_fov(box_data, sensor_position, sensor_rotation_angle, horizontal_fov_min, horizontal_fov_max):
                    continue

                points_in_box = self.box_containing_points(box_data, current_pointcloud)
                if points_in_box >= threshold:
                    box_data['num_lidar_points_in_box_filtered'] = points_in_box  # filtered points
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

    def is_bounding_box_in_sensor_fov(self, box_data, sensor_position, sensor_rotation_angle, horizontal_fov_min, horizontal_fov_max):
        # sensor to bounding box center
        sensor_to_box = box_data['center'][:2] - np.array(sensor_position[:2])
        # angle from sensor to box in world frame
        angle_to_box_world = np.degrees(np.arctan2(sensor_to_box[1], sensor_to_box[0]))
        # calculate minimal angular difference (normalize to [-180, 180])
        angle_diff = (angle_to_box_world - sensor_rotation_angle + 180) % 360 - 180
        # calculate half of the sensor horizontal FOV
        half_fov = (horizontal_fov_max - horizontal_fov_min) / 2
        # cases where FOV spans over 360 or -180/180 boundaries
        if half_fov < 0:
            half_fov += 360
        in_fov = abs(angle_diff) <= half_fov
        return in_fov

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




