#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

#include <pcl/PolygonMesh.h>
#include <pcl/common/point_tests.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/surface/gp3.h>

int main(int argc, char** argv)
{
    if (argc != 9)
    {
        std::cerr
            << "Usage:\n"
            << argv[0]
            << " input.ply output.ply"
            << " search_radius mu max_nn"
            << " max_surface_angle min_angle max_angle\n";

        return EXIT_FAILURE;
    }

    const std::string input_path = argv[1];
    const std::string output_path = argv[2];

    const double search_radius = std::stod(argv[3]);
    const double mu = std::stod(argv[4]);
    const int max_nn = std::stoi(argv[5]);
    const double max_surface_angle_deg = std::stod(argv[6]);
    const double min_angle_deg = std::stod(argv[7]);
    const double max_angle_deg = std::stod(argv[8]);

    using PointNormalT = pcl::PointNormal;

    auto cloud_with_normals =
        pcl::make_shared<pcl::PointCloud<PointNormalT>>();

    if (
        pcl::io::loadPLYFile<PointNormalT>(
            input_path,
            *cloud_with_normals
        ) != 0
    )
    {
        std::cerr
            << "Failed to load point cloud: "
            << input_path
            << "\n";

        return EXIT_FAILURE;
    }

    std::cout
        << "Loaded points: "
        << cloud_with_normals->size()
        << "\n";

    if (cloud_with_normals->empty())
    {
        std::cerr << "Loaded cloud is empty\n";
        return EXIT_FAILURE;
    }

    std::size_t invalid_count = 0;

    for (const auto& point : cloud_with_normals->points)
    {
        if (
            !pcl::isFinite(point)
            || !std::isfinite(point.normal_x)
            || !std::isfinite(point.normal_y)
            || !std::isfinite(point.normal_z)
        )
        {
            ++invalid_count;
        }
    }

    std::cout
        << "Invalid points/normals: "
        << invalid_count
        << "\n";

    if (invalid_count > 0)
    {
        std::cerr
            << "Input contains invalid coordinates or normals\n";

        return EXIT_FAILURE;
    }

    const auto& first = cloud_with_normals->front();

    std::cout
        << "First point: "
        << first.x << ", "
        << first.y << ", "
        << first.z << "\n";

    std::cout
        << "First normal: "
        << first.normal_x << ", "
        << first.normal_y << ", "
        << first.normal_z << "\n";

    auto tree_with_normals =
        pcl::make_shared<pcl::search::KdTree<PointNormalT>>();

    tree_with_normals->setInputCloud(cloud_with_normals);

    pcl::GreedyProjectionTriangulation<PointNormalT> gp3;

    gp3.setSearchRadius(search_radius);
    gp3.setMu(mu);
    gp3.setMaximumNearestNeighbors(max_nn);

    gp3.setMaximumSurfaceAngle(
        max_surface_angle_deg * M_PI / 180.0
    );

    gp3.setMinimumAngle(
        min_angle_deg * M_PI / 180.0
    );

    gp3.setMaximumAngle(
        max_angle_deg * M_PI / 180.0
    );

    gp3.setNormalConsistency(false);

    gp3.setInputCloud(cloud_with_normals);
    gp3.setSearchMethod(tree_with_normals);

    pcl::PolygonMesh mesh;

    const auto start = std::chrono::steady_clock::now();

    gp3.reconstruct(mesh);

    const auto end = std::chrono::steady_clock::now();

    const double runtime_seconds =
        std::chrono::duration<double>(end - start).count();

    std::cout
        << "Generated polygons: "
        << mesh.polygons.size()
        << "\n";

    std::cout
        << "Runtime: "
        << runtime_seconds
        << " seconds\n";

    if (mesh.polygons.empty())
    {
        std::cerr
            << "GP3 generated no polygons. "
            << "Check input normals and parameters.\n";

        return EXIT_FAILURE;
    }

    if (
        pcl::io::savePLYFileBinary(
            output_path,
            mesh
        ) != 0
    )
    {
        std::cerr
            << "Failed to save mesh: "
            << output_path
            << "\n";

        return EXIT_FAILURE;
    }

    std::cout
        << "Saved mesh: "
        << output_path
        << "\n";

    return EXIT_SUCCESS;
}