#include <iostream>
#include <string>

#include <pcl/PolygonMesh.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/io/pcd_io.h>
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
            << " input.pcd output.ply\n";

        return 1;
    }

    const std::string input_file = argv[1];
    const std::string output_file = argv[2];

    using PointT = pcl::PointXYZ;
    using PointNormalT = pcl::PointNormal;

    pcl::PointCloud<PointT>::Ptr cloud(
        new pcl::PointCloud<PointT>
    );

    if (pcl::io::loadPCDFile<PointT>(
            input_file,
            *cloud
        ) != 0)
    {
        std::cerr
            << "Could not read "
            << input_file
            << "\n";

        return 1;
    }

    std::cout
        << "Loaded "
        << cloud->size()
        << " points\n";

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

    normal_estimation.setInputCloud(cloud);
    normal_estimation.setSearchMethod(normal_tree);

    /*
     * Start with 30 cm.
     */
    normal_estimation.setRadiusSearch(0.30);

    pcl::PointCloud<pcl::Normal>::Ptr normals(
        new pcl::PointCloud<pcl::Normal>
    );

    normal_estimation.compute(*normals);

    /*
     * Combine XYZ coordinates and normals.
     */
    pcl::PointCloud<PointNormalT>::Ptr cloud_with_normals(
        new pcl::PointCloud<PointNormalT>
    );

    pcl::concatenateFields(
        *cloud,
        *normals,
        *cloud_with_normals
    );

    pcl::search::KdTree<PointNormalT>::Ptr gp3_tree(
        new pcl::search::KdTree<PointNormalT>
    );

    gp3_tree->setInputCloud(
        cloud_with_normals
    );

    /*
     * Greedy Projection Triangulation.
     */
    pcl::GreedyProjectionTriangulation<
        PointNormalT
    > gp3;

    gp3.setSearchRadius(0.50);
    gp3.setMu(2.5);
    gp3.setMaximumNearestNeighbors(100);

    constexpr double pi = 3.14159265358979323846;

    gp3.setMinimumAngle(
        10.0 * pi / 180.0
    );

    gp3.setMaximumAngle(
        120.0 * pi / 180.0
    );

    gp3.setMaximumSurfaceAngle(
        45.0 * pi / 180.0
    );

    gp3.setNormalConsistency(false);

    gp3.setInputCloud(
        cloud_with_normals
    );

    gp3.setSearchMethod(
        gp3_tree
    );

    pcl::PolygonMesh mesh;

    gp3.reconstruct(mesh);

    std::cout
        << "Generated "
        << mesh.polygons.size()
        << " triangles\n";

    pcl::io::savePLYFileBinary(
        output_file,
        mesh
    );

    std::cout
        << "Saved mesh: "
        << output_file
        << "\n";

    return 0;
}