from pathlib import Path
import argparse

import numpy as np
import open3d as o3d
import tensorflow as tf

from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils.frame_utils import parse_range_image_and_camera_projection
from waymo_open_dataset.utils import transform_utils, range_image_utils

filter_no_label_zone_points = True

def convert_range_image_to_point_cloud(frame,
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
    point_features = []

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

            point_features_tensor = tf.gather_nd(
                range_image_tensor,
                tf.where(range_image_mask)
            )

            point_features.append(
                point_features_tensor.numpy()[:, 1:]
            )

    return points, cp_points, point_features
def extract_point_cloud(frame):
    (
        range_images,
        camera_projections,
        _,
        range_image_top_pose,
    ) = parse_range_image_and_camera_projection(frame)

    points_0, _, features_0 = (
        convert_range_image_to_point_cloud(
            frame,
            range_images,
            camera_projections,
            range_image_top_pose,
            ri_index=0,
            lidar_list=[1, 2, 3, 4, 5],
        )
    )

    points_0 = np.concatenate(
        points_0,
        axis=0,
    )

    features_0 = np.concatenate(
        features_0,
        axis=0,
    )

    points_1, _, features_1 = (
        convert_range_image_to_point_cloud(
            frame,
            range_images,
            camera_projections,
            range_image_top_pose,
            ri_index=1,
            lidar_list=[1, 2, 3, 4, 5],
        )
    )

    points_1 = np.concatenate(
        points_1,
        axis=0,
    )

    features_1 = np.concatenate(
        features_1,
        axis=0,
    )

    points = np.concatenate(
        [points_0, points_1],
        axis=0,
    )

    features = np.concatenate(
        [features_0, features_1],
        axis=0,
    )

    original_pointcloud = np.column_stack(
        (points, features)
    )

    return original_pointcloud

def prepare_point_cloud(
    points_xyz: np.ndarray,
) -> o3d.geometry.PointCloud:
    """
    Create a point cloud and estimate surface normals.
    No voxel downsampling or outlier removal is applied.
    """

    valid = np.isfinite(points_xyz).all(axis=1)
    points_xyz = points_xyz[valid]

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_xyz)

    if len(point_cloud.points) < 3:
        raise ValueError(
            "At least three valid points are needed."
        )

    distances = np.asarray(
        point_cloud.compute_nearest_neighbor_distance()
    )

    valid_distances = distances[
        np.isfinite(distances) & (distances > 0)
    ]

    if len(valid_distances) == 0:
        raise ValueError(
            "Could not estimate point spacing."
        )

    median_spacing = float(np.median(valid_distances))

    normal_radius = max(
        4.0 * median_spacing,
        0.20,
    )

    print(f"Median point spacing: {median_spacing:.4f} m")
    print(f"Normal radius: {normal_radius:.4f} m")

    point_cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius,
            max_nn=50,
        )
    )

    point_cloud.orient_normals_towards_camera_location(
        np.array([0.0, 0.0, 0.0])
    )

    point_cloud.normalize_normals()

    return point_cloud
def reconstruct_ball_pivoting(
    point_cloud: o3d.geometry.PointCloud,
) -> o3d.geometry.TriangleMesh:
    """
    Reconstruct a partial mesh using Ball Pivoting.
    """

    distances = np.asarray(
        point_cloud.compute_nearest_neighbor_distance()
    )

    if len(distances) == 0:
        raise ValueError("Cannot determine point spacing.")

    spacing = float(np.median(distances))

    radii = o3d.utility.DoubleVector(
        [
            1.5 * spacing,
            2.5 * spacing,
            4.0 * spacing,
        ]
    )

    return o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        point_cloud,
        radii,
    )


def reconstruct_poisson(
    point_cloud: o3d.geometry.PointCloud,
    depth: int = 8,
) -> o3d.geometry.TriangleMesh:
    """
    Reconstruct a mesh using screened Poisson reconstruction.
    """

    mesh, densities = (
        o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            point_cloud,
            depth=depth,
            scale=1.1,
            linear_fit=False,
        )
    )

    densities = np.asarray(densities)

    # Remove the least-supported reconstructed regions.
    threshold = np.quantile(densities, 0.02)
    remove_mask = densities < threshold

    mesh.remove_vertices_by_mask(remove_mask)

    return mesh


def reconstruct_alpha_shape(
    point_cloud: o3d.geometry.PointCloud,
    alpha: float = 0.5,
) -> o3d.geometry.TriangleMesh:
    """
    Reconstruct a mesh using an alpha shape.
    """

    tetra_mesh, point_map = (
        o3d.geometry.TetraMesh.create_from_point_cloud(point_cloud)
    )

    return o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        point_cloud,
        alpha,
        tetra_mesh,
        point_map,
    )


def clean_mesh(
    mesh: o3d.geometry.TriangleMesh,
) -> o3d.geometry.TriangleMesh:
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()

    return mesh


def process_tfrecord(
    tfrecord_path: str,
    output_directory: str,
    maximum_frames=None,
) -> None:
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset = tf.data.TFRecordDataset(
        tfrecord_path,
        compression_type="",
    )

    for frame_index, record in enumerate(dataset):
        if maximum_frames is not None and frame_index >= maximum_frames:
            break

        print(f"\nProcessing frame {frame_index:03d}")

        frame = open_dataset.Frame()
        frame.ParseFromString(record.numpy())

        original_pointcloud = extract_point_cloud(frame)
        points_xyz = original_pointcloud[:, :3]

        print(f"Raw points: {len(points_xyz)}")

        point_cloud = prepare_point_cloud(points_xyz)

        print(f"Processed points: {len(point_cloud.points)}")

        frame_directory = output_path / f"frame_{frame_index:03d}"
        frame_directory.mkdir(parents=True, exist_ok=True)

        # Save the original processed frame.
        o3d.io.write_point_cloud(
            str(frame_directory / "point_cloud.ply"),
            point_cloud,
        )

        # -----------------------------------------------------
        # Strategy 1: Ball Pivoting
        # -----------------------------------------------------
        try:
            bpa_mesh = reconstruct_ball_pivoting(point_cloud)
            bpa_mesh = clean_mesh(bpa_mesh)

            o3d.io.write_triangle_mesh(
                str(frame_directory / "mesh_ball_pivoting.ply"),
                bpa_mesh,
            )

            print(
                "Ball Pivoting:",
                len(bpa_mesh.vertices),
                "vertices,",
                len(bpa_mesh.triangles),
                "triangles",
            )
        except Exception as error:
            print(f"Ball Pivoting failed: {error}")

        # -----------------------------------------------------
        # Strategy 2: Poisson
        # -----------------------------------------------------
        try:
            poisson_mesh = reconstruct_poisson(
                point_cloud,
                depth=8,
            )
            poisson_mesh = clean_mesh(poisson_mesh)

            o3d.io.write_triangle_mesh(
                str(frame_directory / "mesh_poisson.ply"),
                poisson_mesh,
            )

            print(
                "Poisson:",
                len(poisson_mesh.vertices),
                "vertices,",
                len(poisson_mesh.triangles),
                "triangles",
            )
        except Exception as error:
            print(f"Poisson failed: {error}")

        # Alpha shapes may become expensive for an entire Waymo cloud.
        # Enable this later on cropped or more strongly downsampled clouds.
        #
        # try:
        #     alpha_mesh = reconstruct_alpha_shape(
        #         point_cloud,
        #         alpha=0.5,
        #     )
        #     alpha_mesh = clean_mesh(alpha_mesh)
        #
        #     o3d.io.write_triangle_mesh(
        #         str(frame_directory / "mesh_alpha_shape.ply"),
        #         alpha_mesh,
        #     )
        # except Exception as error:
        #     print(f"Alpha shape failed: {error}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tfrecord",
        required=True,
        help="Path to one Waymo TFRecord.",
    )
    parser.add_argument(
        "--output",
        default="surface_reconstruction_results",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    process_tfrecord(
        tfrecord_path=args.tfrecord,
        output_directory=args.output,
        maximum_frames=args.max_frames,
    )
    
    
    
## terminal command

# python reconstruct_waymo.py \
#     --tfrecord /data/waymo/raw_data/segment-898816942644052013_20_000_40_000_with_camera_labels.tfrecord \
#     --output reconstruction_test \
#     --max_frames 3