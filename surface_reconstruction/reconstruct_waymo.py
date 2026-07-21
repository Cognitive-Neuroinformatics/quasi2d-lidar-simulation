from pathlib import Path
import argparse

import numpy as np
import open3d as o3d
import tensorflow as tf

from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.utils import frame_utils


def extract_waymo_point_cloud(frame: dataset_pb2.Frame) -> np.ndarray:
    """
    Convert all lidar returns in one Waymo frame into one merged XYZ cloud
    in the Waymo vehicle coordinate system.
    """

    (
        range_images,
        camera_projections,
        segmentation_labels,
        range_image_top_pose,
    ) = frame_utils.parse_range_image_and_camera_projection(frame)

    points, _ = frame_utils.convert_range_image_to_point_cloud(
        frame=frame,
        range_images=range_images,
        camera_projections=camera_projections,
        range_image_top_pose=range_image_top_pose,
        ri_index=0,
    )

    # points is a list: one point cloud for each Waymo lidar.
    merged_points = np.concatenate(points, axis=0)

    return merged_points[:, :3].astype(np.float64)


def prepare_point_cloud(
    points_xyz: np.ndarray,
    voxel_size: float = 0.10,
) -> o3d.geometry.PointCloud:
    """
    Create, clean and estimate normals for an Open3D point cloud.
    """

    valid = np.isfinite(points_xyz).all(axis=1)
    points_xyz = points_xyz[valid]

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_xyz)

    # Optional downsampling.
    point_cloud = point_cloud.voxel_down_sample(voxel_size)

    # Remove isolated points.
    if len(point_cloud.points) >= 30:
        point_cloud, _ = point_cloud.remove_statistical_outlier(
            nb_neighbors=20,
            std_ratio=2.0,
        )

    # Normals are required by Ball Pivoting and Poisson.
    normal_radius = max(voxel_size * 4.0, 0.30)

    point_cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius,
            max_nn=50,
        )
    )

    # Approximate the Waymo vehicle origin as the viewpoint.
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
    maximum_frames = None,
    voxel_size = 0.10,
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

        frame = dataset_pb2.Frame()
        frame.ParseFromString(record.numpy())

        points_xyz = extract_waymo_point_cloud(frame)

        print(f"Raw points: {len(points_xyz)}")

        point_cloud = prepare_point_cloud(
            points_xyz,
            voxel_size=voxel_size,
        )

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
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.10,
    )

    args = parser.parse_args()

    process_tfrecord(
        tfrecord_path=args.tfrecord,
        output_directory=args.output,
        maximum_frames=args.max_frames,
        voxel_size=args.voxel_size,
    )
    
    
    
## terminal command

# python reconstruct_waymo.py \
#     --tfrecord /data/waymo/raw_data/segment-898816942644052013_20_000_40_000_with_camera_labels.tfrecord \
#     --output reconstruction_test \
#     --max_frames 3 \
#     --voxel_size 0.10