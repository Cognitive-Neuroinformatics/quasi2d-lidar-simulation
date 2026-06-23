import os
import yaml
import numpy as np
import tensorflow as tf
import transformations as tf_transformations 
from pathlib import Path
import tqdm
import argparse
from multiprocessing import Pool, cpu_count
import cv2 as cv
import random
from collections import deque

from waymo_open_dataset.utils.frame_utils import parse_range_image_and_camera_projection
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils

from pointcloud_transformer.pointcloud_transformer import PointCloudTransformer
from transform_utils.transform_scan import TransformScan3D, TransformScan2D
from utils import gt_sampling_utils
from fusion.point_fusion import fuse_multi_with_dt
from labels.annotation_transforms import transform_annos_baselink_to_sensor, transform_annos_baselink_to_baselink
from labels.label_writer import write_openpcdet_label_file_multiframe
from transform_utils.math_utils import make_4x4_from_xy_yaw

filter_no_label_zone_points = True

class WaymoDataProcessor:
    def __init__(
        self,
        config_path,
        load_dir,
        save_dir,
        sensor,
        prefix,
        save_outputs=None,
        selected_sensors=None,
        num_proc=28,
        transformation_mode="localsensor"
    ):    
        # turn on eager execution for older tensorflow versions
        if int(tf.__version__.split('.')[0]) < 2:
            tf.enable_eager_execution()

        self.num_proc = num_proc if num_proc is not None else cpu_count() 
        total_cpus = cpu_count()
        print(f"[INFO] Using {self.num_proc} worker processes out of {total_cpus} available CPUs")
        
        # Directories
        self.config_path= config_path
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)
            
        self.load_dir = load_dir
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)               
        self.prefix   = prefix
        
        # loading config parameters
        
        self.x_size = self.cfg["bev"]["x_size"]
        self.y_size = self.cfg["bev"]["y_size"]
        self.voxel_size = self.cfg["bev"]["voxel_size"]
        self.bev_padding = self.cfg["bev"]["bev_padding"]
        self.data_percent = self.cfg["dataset"]["data_percent"]
        self.sets_dir = self.cfg["dataset"]["sets_dir"]
        self.z_min = self.cfg["bev"]["z_min"]       
        self.z_max = self.cfg["bev"]["z_max"]

        self.save_outputs = set(save_outputs)
        self.selected_sensors = (set(selected_sensors) if selected_sensors is not None else None)
        
        self.resolution = float(self.voxel_size)
        self.load_tfrecords_by_prefix()
        self.sensor = sensor
        self.transformation_mode = transformation_mode
        self.transformer = PointCloudTransformer(voxel_size=self.voxel_size, ground_removal_method=None)
        self.transformer3D = TransformScan3D(self.sensor)
        self.transformer2D = TransformScan2D(self.sensor, self.x_size, self.y_size, self.z_min, self.z_max, self.resolution)
        

    def load_tfrecords_by_prefix(self):
        # filename sets, may need to be changed later
        if self.prefix == 'train':
            set_file = os.path.join(self.sets_dir, 'train.txt')
        elif self.prefix == 'val':
            set_file = os.path.join(self.sets_dir, 'val.txt')
        elif self.prefix == 'test':
            set_file = os.path.join(self.sets_dir, 'test.txt')
        else:
            raise ValueError("Unknown prefix '{self.prefix}'. Expected one of "
                            "train, train_missing, val, val20, test, test_small.")

        if not os.path.exists(set_file):
            raise FileNotFoundError(f"No file found for prefix {self.prefix} in {self.sets_dir}")

        with open(set_file, 'r') as file:
            all_files = [line.strip() for line in file.readlines()]
            selected_files = random.sample(all_files, int(len(all_files) * self.data_percent / 100)) # selected_files are shuffled randomly

        self.tfrecord_pathnames = [os.path.join(self.load_dir, file_name) for file_name in selected_files]
        self.tfrecord_pathnames = sorted(filter(os.path.exists, self.tfrecord_pathnames))


    def process_data(self):
        with Pool(processes=self.num_proc) as pool:
            list(tqdm.tqdm(pool.imap(self.process_file, self.tfrecord_pathnames), total=len(self.tfrecord_pathnames), desc='Processing tfrecord files ...'))

    def process_file(self, tfrecord_file):
        os.environ['CUDA_VISIBLE_DEVICES'] = ''  # temp fix for tensorflow mishandling multiprocessing
        dataset = tf.data.TFRecordDataset(tfrecord_file, compression_type='')
        sequence_name = Path(tfrecord_file).stem

        for frame_index, data in enumerate(dataset):
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))

            original_pointcloud = self.extract_point_cloud(frame)
            transformed_pointclouds, transformation_metadata = self.transform_point_clouds(original_pointcloud)
            if not any(pc.size > 0 for pc in transformed_pointclouds):
                continue
            else:
                if self.transformation_mode == 'localsensor':
                    
                    self.process_frames_local_sensor_with_multiframe_early_fusion(frame, transformed_pointclouds,
                                        transformation_metadata, sequence_name, frame_index, openpcdet = True)
                else:
                    raise ValueError(f"Unknown transformation_mode: {self.transformation_mode}")


    def transform_point_clouds(self, original_pointcloud):

        ## This function bypasses the missing sensor logic. currently implemented in PC for swifter calculations
        params = self.get_transformation_params()
        transformed_pointclouds = []
        transformation_metadata = []
        
        for param_set in params:
            sensor_name, start_point, distance, horizontal_angle_min, horizontal_angle_max, horizontal_rays, horizontal_increment, vertical_angles, rotation_angle = param_set
            transformed_pointcloud = self.transformer.transform_point_cloud(
                self.sensor,
                original_pointcloud,
                start_point,
                distance,
                horizontal_angle_min,
                horizontal_angle_max,
                horizontal_rays,
                vertical_angles,
                rotation_angle
            )
            transformed_pointclouds.append(transformed_pointcloud)
            # metadata for transforming multiple sensors in BEV processing
            transformation_metadata.append({
                "sensor_name": sensor_name,
                "start_point": start_point,
                "rotation_angle": rotation_angle,
                "horizontal_fov_min": horizontal_angle_min,
                "horizontal_fov_max": horizontal_angle_max,
            })

        return transformed_pointclouds, transformation_metadata

   
    def process_frames_local_sensor_with_multiframe_early_fusion(
        self,
        frame,
        transformed_pointclouds,
        transformation_metadata,
        sequence_name,
        frame_index,
        openpcdet=True
    ):

        vehicle_pose = np.array(frame.pose.transform).reshape(4, 4)
        ts_curr = int(frame.timestamp_micros) # current timestamp
        # store up to 3 previous frames => total up to 4 frames fused
        if not hasattr(self, "previous_buffer"):
            self.previous_buffer = {}  # sensor_i -> deque(maxlen=3)

        

        for i, (transformed_pointcloud, metadata) in enumerate(zip(transformed_pointclouds, transformation_metadata)):
            if transformed_pointcloud.size == 0:
                continue
            if not openpcdet:
                continue

            sensor_name = metadata["sensor_name"]
            sensor_dir = self.save_dir / sensor_name
            
            npy_file_name = f"{sequence_name}_{sensor_name}_{frame_index:03d}.npy"
            img_file_name = f"{sequence_name}_{sensor_name}_{frame_index:03d}.png"
            yaw_deg = metadata["rotation_angle"]

            filtered_bounding_boxes = self.transformer.filter_bounding_boxes(
                frame,
                transformed_pointcloud,
                sensor_position=metadata["start_point"],
                sensor_rotation_angle=yaw_deg,
                horizontal_fov_min=metadata["horizontal_fov_min"],
                horizontal_fov_max=metadata["horizontal_fov_max"],
                threshold=4
            )


            # for 1f, 2f and 4f point clouds, directory creation:
            points_dir_1f = sensor_dir / "points_concat_1f"
            points_dir_2f = sensor_dir / "points_concat_2f"
            points_dir_4f = sensor_dir / "points_concat_4f"
            
            
            # for 2d merged images, directory creation
            image_dir_height = sensor_dir / "images_height"
            image_dir_overlap = sensor_dir / "images_overlap"
            image_dir_overlap_height = sensor_dir / "images_overlap_height"

            if "images_height" in self.save_outputs:
                image_dir_height.mkdir(parents=True, exist_ok=True)

            if "images_overlap" in self.save_outputs:
                image_dir_overlap.mkdir(parents=True, exist_ok=True)

            if "images_overlap_height" in self.save_outputs:
                image_dir_overlap_height.mkdir(parents=True, exist_ok=True)

            if "points_1f" in self.save_outputs:
                points_dir_1f.mkdir(parents=True, exist_ok=True)

            if "points_2f" in self.save_outputs:
                points_dir_2f.mkdir(parents=True, exist_ok=True)

            if "points_4f" in self.save_outputs:
                points_dir_4f.mkdir(parents=True, exist_ok=True)

            label_subdirs = []

            if "labels" in self.save_outputs:
                label_subdirs.append("labels")

            if "labels_2f" in self.save_outputs:
                label_subdirs.append("labels_2f")

            if "labels_4f" in self.save_outputs:
                label_subdirs.append("labels_4f")


            
            # 1) write label file in SENSOR frame, but RETURN annos_with_track in BASELINK 
            need_labels = len(label_subdirs) > 0 
            if need_labels:
                annos_cur_baselink = write_openpcdet_label_file_multiframe(
                    self.transformer,
                    sensor_name,
                    sequence_name,
                    filtered_bounding_boxes,
                    frame_index,
                    sensor_dir,
                    bev_padding=self.bev_padding,
                    metadata=metadata,
                    transformation_mode="localsensor",
                    return_with_track=True,
                    label_subdirs=label_subdirs
                )
            else:
                annos_cur_baselink = None

            # 2) current points -> sensor frame
            pc_curr_baselink = transformed_pointcloud
            pc_curr_sensor = self.transformer.transform_pc_to_sensor_frame(
                pc_curr_baselink, metadata, bev_padding=self.bev_padding
            )

            # 3) history buffer
            buf = self.previous_buffer.get(i)
            if buf is None:
                buf = deque(maxlen=3)
                self.previous_buffer[i] = buf

            # 4) precompute sensor transform for THIS frame
            yaw_rad = np.deg2rad(metadata["rotation_angle"])
            T_baselink_from_sensor, T_sensor_from_baselink = self.transformer.get_sensor_transforms(metadata, self.bev_padding)

            # 5) build previous pointclouds in current sensor frame
            prev_sensor_list = []
            prev_ts_list = []
            for item in reversed(buf):  # t-1, t-2, t-3
                prev_pose = item["pose"]
                prev_pc_baselink = item["pc"]
                prev_ts = int(item.get("ts", ts_curr))

                T_curr_from_prev = self.transformer3D.relative_ego_motion(
                    prev_pose, vehicle_pose, pose_convention="vehicle_to_world"
                )
                prev_in_curr_baselink = self.transformer3D.transform_points_4x4(prev_pc_baselink, T_curr_from_prev)
                prev_in_curr_sensor = self.transformer.transform_pc_to_sensor_frame(
                    prev_in_curr_baselink, metadata, bev_padding=self.bev_padding
                )
                prev_sensor_list.append(prev_in_curr_sensor)  # [t-1, t-2, t-3]
                prev_ts_list.append(prev_ts)  # [t-1, t-2, t-3]


            if "points_1f" in self.save_outputs:
                
                fused_1f = pc_curr_sensor.astype(np.float32, copy=False)[:, :4]
                np.save(points_dir_1f / npy_file_name, fused_1f)

            if "points_2f" in self.save_outputs:
                fused_2f = fuse_multi_with_dt(
                    pc_curr_sensor,
                    ts_curr,
                    prev_sensor_list,
                    prev_ts_list,
                    num_frames=2
                )
                np.save(points_dir_2f / npy_file_name, fused_2f)

            if "points_4f" in self.save_outputs:
                fused_4f = fuse_multi_with_dt(
                    pc_curr_sensor,
                    ts_curr,
                    prev_sensor_list,
                    prev_ts_list,
                    num_frames=4
                )
                np.save(points_dir_4f / npy_file_name, fused_4f)
            
            if (
                "images_height" in self.save_outputs
                or "images_overlap" in self.save_outputs
                or "images_overlap_height" in self.save_outputs
            ):
                if len(buf) > 0:
                    prev_vehicle_pose = buf[-1]["pose"]
                    prev_pose = make_4x4_from_xy_yaw(
                        prev_vehicle_pose[0, 3],
                        prev_vehicle_pose[1, 3],
                        tf_transformations.euler_from_matrix(prev_vehicle_pose)[2]
                    )
                    prev_pointcloud = buf[-1]["pc_sensor"]
                    
                else:
                    prev_pose = None
                    prev_pointcloud = None


                T_sensor_4x4_vehicle = make_4x4_from_xy_yaw(
                    vehicle_pose[0, 3],
                    vehicle_pose[1, 3],
                    tf_transformations.euler_from_matrix(vehicle_pose)[2]
                )

                if "images_height" in self.save_outputs:
                    img_height = self.transformer2D.transform_odom(
                        pc_curr_sensor[:, :3],
                        T_sensor_4x4_vehicle,
                        method="height",
                        prev_pose=None,
                        prev_pointcloud=None,
                        T_baselink_from_sensor=None
                    )
                    cv.imwrite(str(image_dir_height / img_file_name), img_height)

                if "images_overlap" in self.save_outputs:
                    img_overlap = self.transformer2D.transform_odom(
                        pc_curr_sensor[:, :3],
                        T_sensor_4x4_vehicle,
                        method="overlap",
                        prev_pose=prev_pose,
                        prev_pointcloud=prev_pointcloud,
                        T_baselink_from_sensor=T_baselink_from_sensor
                    )
                    cv.imwrite(str(image_dir_overlap / img_file_name), img_overlap)

                if "images_overlap_height" in self.save_outputs:
                    img_overlap_height = self.transformer2D.transform_odom(
                        pc_curr_sensor[:, :3],
                        T_sensor_4x4_vehicle,
                        method="overlap_height",
                        prev_pose=prev_pose,
                        prev_pointcloud=prev_pointcloud,
                        T_baselink_from_sensor=T_baselink_from_sensor
                    )
                    cv.imwrite(str(image_dir_overlap_height / img_file_name), img_overlap_height)


            # ============================================================
            # GT for 4 frames, ego-motion compensate (baselink),
            # then transform to sensor, then store in GT tracker.
            # frame_id 0 = current, 1 = t-1, 2 = t-2, 3 = t-3
            # ============================================================

            need_gt_2f = "gt_2f" in self.save_outputs
            need_gt_4f = "gt_4f" in self.save_outputs

            if need_gt_2f or need_gt_4f:
                gt_sampler_dir = sensor_dir / "labels_gt_sampler_json"
                gt_track_writer = gt_sampling_utils.GTTrackJSONWriter(gt_sampler_dir)
                
                
                annos_list_sensor_in_cur = [None, None, None, None]

                annos_list_sensor_in_cur[0] = transform_annos_baselink_to_sensor(
                    annos_cur_baselink,
                    T_sensor_from_baselink,
                    yaw_rad
                )

                for k, item in enumerate(reversed(buf), start=1):
                    prev_pose = item["pose"]
                    prev_annos_baselink = item["gt"]

                    T_curr_from_prev = self.transformer3D.relative_ego_motion(
                        prev_pose,
                        vehicle_pose,
                        pose_convention="vehicle_to_world"
                    )

                    prev_annos_in_curr_baselink = transform_annos_baselink_to_baselink(
                        prev_annos_baselink,
                        T_curr_from_prev
                    )

                    prev_annos_in_curr_sensor = transform_annos_baselink_to_sensor(
                        prev_annos_in_curr_baselink,
                        T_sensor_from_baselink,
                        yaw_rad
                    )

                    if k < 4:
                        annos_list_sensor_in_cur[k] = prev_annos_in_curr_sensor

                if need_gt_2f:
                    gt_track_writer.update_multiframe(
                        sequence_name=sequence_name,
                        sensor_i=sensor_name,
                        frame_index=frame_index,
                        annos_list_in_cur_with_track=annos_list_sensor_in_cur[:2]
                    )

                if need_gt_4f:
                    gt_track_writer.update_multiframe(
                        sequence_name=sequence_name,
                        sensor_i=sensor_name,
                        frame_index=frame_index,
                        annos_list_in_cur_with_track=annos_list_sensor_in_cur
                    )


            # 6) update history buffer (store ORIGINAL baselink-frame pc + BASELINK annos)
            buf.append({
                "pose": vehicle_pose,
                "pc": pc_curr_baselink.copy(),
                "pc_sensor": pc_curr_sensor.copy(),
                "gt": annos_cur_baselink,  # baselink for future ego-motion
                "ts": int(frame.timestamp_micros),  # storing current timestamp
            })

                             
    def extract_point_cloud(self, frame):
        range_images, camera_projections, _, range_image_top_pose = parse_range_image_and_camera_projection(frame)
        points_0, cp_points_0, intensity_0 = self.convert_range_image_to_point_cloud(
            frame,
            range_images,
            camera_projections,
            range_image_top_pose,
            ri_index=0,
            lidar_list=[1,2,3,4,5]
        )
        points_0 = np.concatenate(points_0, axis=0)
        intensity_0 = np.concatenate(intensity_0, axis=0)

        points_1, cp_points_1, intensity_1 = self.convert_range_image_to_point_cloud(
            frame,
            range_images,
            camera_projections,
            range_image_top_pose,
            ri_index=1,
            lidar_list=[1,2,3,4,5]
        )
        points_1 = np.concatenate(points_1, axis=0)
        intensity_1 = np.concatenate(intensity_1, axis=0)

        points = np.concatenate([points_0, points_1], axis=0)
        intensity = np.concatenate([intensity_0, intensity_1], axis=0)
        original_pointcloud = np.column_stack((points, intensity))
        return original_pointcloud
    
    def get_transformation_params(self):
        sensors_cfg = self.cfg["sensors"]

        if self.sensor not in sensors_cfg:
            raise ValueError(
                f"Unknown sensor '{self.sensor}'. Available sensors: {list(sensors_cfg.keys())}"
            )

        sensor_cfg = sensors_cfg[self.sensor]
        sensor_positions = sensor_cfg["positions"]
        common = sensor_cfg["common"]

        params = []

        for sensor_pos in sensor_positions:
            
            sensor_name = sensor_pos["name"]
            
            if (
                self.selected_sensors is not None
                and sensor_name not in self.selected_sensors
            ):
                continue
            
            start_point = tuple(sensor_pos["start_point"])
            rotation_angle = float(sensor_pos["rotation_angle"])

            horizontal_angle_min = common["horizontal_angle_min"] + rotation_angle
            horizontal_angle_max = common["horizontal_angle_max"] + rotation_angle

            params.append((
                sensor_pos["name"],
                start_point,
                common["distance"],
                horizontal_angle_min,
                horizontal_angle_max,
                common["horizontal_rays"],
                common["horizontal_increment"],
                common["vertical_angles"],
                rotation_angle
            ))

        return params
    
    def convert_range_image_to_point_cloud(self,
                                           frame,
                                           range_images,
                                           camera_projections,
                                           range_image_top_pose,
                                           ri_index=0,
                                           lidar_list=[1,2,3,4,5]):
        """Convert range images to point cloud.
        Args:
          frame: open dataset frame
           range_images: A dict of {laser_name, [range_image_first_return,
             range_image_second_return]}.
           camera_projections: A dict of {laser_name,
             [camera_projection_from_first_return,
             camera_projection_from_second_return]}.
          range_image_top_pose: range image pixel pose for top lidar.
          ri_index: 0 for the first return, 1 for the second return.
          lidar_list: List of lidar sensors to convert, default all = [1,2,3,4,5]
        Returns:
          points: {[N, 3]} list of 3d lidar points of length 5 (number of lidars).
          cp_points: {[N, 6]} list of camera projections of length 5
            (number of lidars).
        """
        calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
        points = []
        cp_points = []
        intensity = []

        frame_pose = tf.convert_to_tensor(
            value=np.reshape(np.array(frame.pose.transform), [4, 4]))
        # [H, W, 6]
        range_image_top_pose_tensor = tf.reshape(
            tf.convert_to_tensor(value=range_image_top_pose.data),
            range_image_top_pose.shape.dims)
        # [H, W, 3, 3]
        range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(
            range_image_top_pose_tensor[..., 0], range_image_top_pose_tensor[..., 1],
            range_image_top_pose_tensor[..., 2])
        range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
        range_image_top_pose_tensor = transform_utils.get_transform(
            range_image_top_pose_tensor_rotation,
            range_image_top_pose_tensor_translation)

        for c in calibrations:
            if c.name in lidar_list:
                #print(c.name)
                range_image = range_images[c.name][ri_index]
                if len(c.beam_inclinations) == 0:  # pylint: disable=g-explicit-length-test
                    beam_inclinations = range_image_utils.compute_inclination(
                        tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
                        height=range_image.shape.dims[0])
                else:
                    beam_inclinations = tf.constant(c.beam_inclinations)

                beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
                extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])


                range_image_tensor = tf.reshape(
                    tf.convert_to_tensor(value=range_image.data), range_image.shape.dims)


                pixel_pose_local = None
                frame_pose_local = None
                if c.name == open_dataset.LaserName.TOP:
                    pixel_pose_local = range_image_top_pose_tensor
                    pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
                    frame_pose_local = tf.expand_dims(frame_pose, axis=0)
                range_image_mask = range_image_tensor[..., 0] > 0

                # No Label Zone
                if filter_no_label_zone_points:
                    nlz_mask = range_image_tensor[..., 3] != 1.0  # 1.0: in NLZ
                    # print(range_image_tensor[range_image_tensor[..., 3] == 1.0])
                    range_image_mask = range_image_mask & nlz_mask

                range_image_cartesian = range_image_utils.extract_point_cloud_from_range_image(
                    tf.expand_dims(range_image_tensor[..., 0], axis=0),
                    tf.expand_dims(extrinsic, axis=0),
                    tf.expand_dims(tf.convert_to_tensor(value=beam_inclinations), axis=0),
                    pixel_pose=pixel_pose_local,
                    frame_pose=frame_pose_local)

                range_image_polar = range_image_utils.compute_range_image_polar(
                    tf.expand_dims(range_image_tensor[..., 0], axis=0),
                    tf.expand_dims(extrinsic, axis=0),
                    tf.expand_dims(tf.convert_to_tensor(value=beam_inclinations), axis=0))
                range_image_polar = tf.squeeze(range_image_polar, axis=0)

                range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)
                points_tensor = tf.gather_nd(range_image_cartesian,
                                            tf.compat.v1.where(range_image_mask))

                cp = camera_projections[c.name][ri_index]
                cp_tensor = tf.reshape(tf.convert_to_tensor(value=cp.data), cp.shape.dims)
                cp_points_tensor = tf.gather_nd(cp_tensor,
                                                tf.compat.v1.where(range_image_mask))
                points.append(points_tensor.numpy())
                cp_points.append(cp_points_tensor.numpy())

                intensity_tensor = tf.gather_nd(range_image_tensor,
                                                tf.where(range_image_mask))
                intensity.append(intensity_tensor.numpy()[:, 1])

        return points, cp_points, intensity    
    
    
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Waymo Open Dataset Processor')
    parser.add_argument('--config',type=str,required=True,help='Path to YAML config file')
    parser.add_argument('--load_dir', type=str, required=True, help='Directory to load Waymo Open Dataset tfrecords')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save processed data')
    parser.add_argument('--sensor', type=str, required=True, help='Sensor to choose: scala, scala2')
    parser.add_argument("--prefix", type=str, default="train", choices=["train", "val", "test"], help="Dataset split to process.")
    parser.add_argument('--run_train_and_val', action='store_true', help='Run transformation for both train and validation sequentially.')
    parser.add_argument('--num_proc', type=int, default=1, help='Number of processes to use for multiprocessing')
    parser.add_argument("--save_outputs", nargs="+", default=["labels", "images_overlap", "images_height", "images_overlap_height", "points_1f", "points_2f", "points_4f", "gt_2f", "gt_4f"],
    choices=["labels", "labels_2f", "labels_4f", "images_overlap", "images_height", "images_overlap_height", "points_1f", "points_2f", "points_4f", "gt_2f", "gt_4f"],
    help="Choose outputs to save.")
    parser.add_argument("--selected_sensors", nargs="+", default=None, 
        help=(
        "Optional subset of sensor names to process. "
        "Example: front_left front_center rear_center"
    ))
    parser.add_argument('--transformation_mode', type=str, default='localsensor',
                        choices=['localsensor'],
                        help='Transformation method to use: "localsensor" for local sensor positioning. Currently this code supports only this mode.')
    args = parser.parse_args()
    
    if args.run_train_and_val:
        # Run TRAIN
        print("[INFO] Starting transformation for training data...")
        processor = WaymoDataProcessor(
            args.config, args.load_dir, os.path.join(args.save_dir, "training"), args.sensor, 'train', save_outputs=args.save_outputs, selected_sensors=args.selected_sensors, num_proc=args.num_proc, transformation_mode=args.transformation_mode
        )
        processor.process_data()

        print("\n\nTransformation finished for training data, now switching to validation data\n\n")

        
        # Run VAL
        print("[INFO] Starting transformation for validation data...")
        processor = WaymoDataProcessor(
            args.config, args.load_dir, os.path.join(args.save_dir, "validation"), args.sensor, 'val', save_outputs=args.save_outputs, selected_sensors=args.selected_sensors,num_proc=args.num_proc, transformation_mode=args.transformation_mode
        )
        processor.process_data()
    else:
        # Original single-prefix behavior remains unchanged prefix: train or val
        if args.prefix == "train":
            save_dir = os.path.join(args.save_dir, "training")
        elif args.prefix == "val":
            save_dir = os.path.join(args.save_dir, "validation")
        elif args.prefix == "test":
            save_dir = os.path.join(args.save_dir, "testing")
        else:
            raise ValueError(f"Unknown prefix: {args.prefix}")

        processor = WaymoDataProcessor(
            config_path=args.config,
            load_dir=args.load_dir,
            save_dir=save_dir,
            sensor=args.sensor,
            prefix=args.prefix,
            save_outputs=args.save_outputs,
            selected_sensors=args.selected_sensors,
            num_proc=args.num_proc,
            transformation_mode=args.transformation_mode,
        )
        processor.process_data()

