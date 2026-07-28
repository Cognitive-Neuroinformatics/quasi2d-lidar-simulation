#include <iostream>
#include <string>

#include <pcl/PolygonMesh.h>
#include <pcl/common/io.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/surface/gp3.h>


int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cerr
            << "Usage: "
            << argv[0]
            << " input.ply output_mesh.ply\n";

        return 1;
    }

    const std::string input_file = argv[1];
    const std::string output_file = argv[2];

    using PointT = pcl::PointXYZ;
    using PointNormalT = pcl::PointNormal;

    pcl::PointCloud<PointT>::Ptr cloud(
        new pcl::PointCloud<PointT>
    );

    /*
     * Load the Gaussian Splatting PLY.
     *
     * Only x, y and z are loaded into PointXYZ.
     * Additional Gaussian properties are ignored.
     */
    const int load_result =
        pcl::io::loadPLYFile<PointT>(
            input_file,
            *cloud
        );

    if (load_result < 0)
    {
        std::cerr
            << "Could not read PLY file: "
            << input_file
            << "\n";

        return 1;
    }

    std::cout
        << "Loaded "
        << cloud->size()
        << " Gaussian centres\n";

    if (cloud->empty())
    {
        std::cerr
            << "The input cloud is empty.\n";

        return 1;
    }

    /*
     * Remove invalid coordinates.
     */
    pcl::PointCloud<PointT>::Ptr valid_cloud(
        new pcl::PointCloud<PointT>
    );

    valid_cloud->reserve(cloud->size());

    for (const PointT& point : cloud->points)
    {
        if (pcl::isFinite(point))
        {
            valid_cloud->push_back(point);
        }
    }

    valid_cloud->width =
        static_cast<std::uint32_t>(
            valid_cloud->size()
        );

    valid_cloud->height = 1;
    valid_cloud->is_dense = true;

    std::cout
        << "Valid Gaussian centres: "
        << valid_cloud->size()
        << "\n";

    if (valid_cloud->size() < 3)
    {
        std::cerr
            << "Not enough valid points for triangulation.\n";

        return 1;
    }

    /*
     * Estimate normals.
     */
    pcl::NormalEstimationOMP<
        PointT,
        pcl::Normal
    > normal_estimation;

    pcl::search::KdTree<PointT>::Ptr normal_tree(
        new pcl::search::KdTree<PointT>
    );

    normal_estimation.setInputCloud(
        valid_cloud
    );

    normal_estimation.setSearchMethod(
        normal_tree
    );

    /*
     * Normal estimation radius in metres.
     */
    normal_estimation.setRadiusSearch(
        0.40
    );

    pcl::PointCloud<pcl::Normal>::Ptr normals(
        new pcl::PointCloud<pcl::Normal>
    );

    normal_estimation.compute(
        *normals
    );

    if (normals->size() != valid_cloud->size())
    {
        std::cerr
            << "Normal count does not match point count.\n";

        return 1;
    }

    /*
     * Combine XYZ coordinates and normals.
     */
    pcl::PointCloud<PointNormalT>::Ptr cloud_with_normals(
        new pcl::PointCloud<PointNormalT>
    );

    pcl::concatenateFields(
        *valid_cloud,
        *normals,
        *cloud_with_normals
    );

    /*
     * Remove points whose normals could not be estimated.
     */
    pcl::PointCloud<PointNormalT>::Ptr valid_cloud_with_normals(
        new pcl::PointCloud<PointNormalT>
    );

    valid_cloud_with_normals->reserve(
        cloud_with_normals->size()
    );

    for (const PointNormalT& point : cloud_with_normals->points)
    {
        if (pcl::isFinite(point) &&
            std::isfinite(point.normal_x) &&
            std::isfinite(point.normal_y) &&
            std::isfinite(point.normal_z))
        {
            valid_cloud_with_normals->push_back(
                point
            );
        }
    }

    valid_cloud_with_normals->width =
        static_cast<std::uint32_t>(
            valid_cloud_with_normals->size()
        );

    valid_cloud_with_normals->height = 1;
    valid_cloud_with_normals->is_dense = true;

    std::cout
        << "Points with valid normals: "
        << valid_cloud_with_normals->size()
        << "\n";

    if (valid_cloud_with_normals->size() < 3)
    {
        std::cerr
            << "Not enough points with valid normals.\n";

        return 1;
    }

    pcl::search::KdTree<PointNormalT>::Ptr gp3_tree(
        new pcl::search::KdTree<PointNormalT>
    );

    gp3_tree->setInputCloud(
        valid_cloud_with_normals
    );

    /*
     * Greedy Projection Triangulation.
     */
    pcl::GreedyProjectionTriangulation<
        PointNormalT
    > gp3;

    /*
     * Maximum edge/search distance in metres.
     */
    gp3.setSearchRadius(
        0.60
    );

    /*
     * Multiplication factor applied to the local
     * nearest-neighbour distance.
     */
    gp3.setMu(
        2.5
    );

    gp3.setMaximumNearestNeighbors(
        300
    );

    constexpr double pi =
        3.14159265358979323846;

    gp3.setMinimumAngle(
        10.0 * pi / 180.0
    );

    gp3.setMaximumAngle(
        120.0 * pi / 180.0
    );

    gp3.setMaximumSurfaceAngle(
        45.0 * pi / 180.0
    );

    /*
     * Set this to true only when normals are
     * consistently oriented.
     */
    gp3.setNormalConsistency(
        false
    );

    gp3.setInputCloud(
        valid_cloud_with_normals
    );

    gp3.setSearchMethod(
        gp3_tree
    );

    pcl::PolygonMesh mesh;

    gp3.reconstruct(
        mesh
    );

    std::cout
        << "Generated "
        << mesh.polygons.size()
        << " triangles\n";

    if (mesh.polygons.empty())
    {
        std::cerr
            << "No triangles were generated.\n"
            << "Try increasing the normal radius or "
            << "GP3 search radius.\n";

        return 1;
    }

    const int save_result =
        pcl::io::savePLYFileBinary(
            output_file,
            mesh
        );

    if (save_result < 0)
    {
        std::cerr
            << "Could not save mesh to: "
            << output_file
            << "\n";

        return 1;
    }

    std::cout
        << "Saved mesh: "
        << output_file
        << "\n";

    return 0;
}