#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

#include <pcl/PolygonMesh.h>
#include <pcl/common/point_tests.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/surface/marching_cubes_hoppe.h>


int main(int argc, char** argv)
{
    if (argc != 9)
    {
        std::cerr
            << "Usage:\n"
            << argv[0]
            << " input.ply output.ply"
            << " grid_x grid_y grid_z"
            << " iso_level percentage_extend\n";

        return EXIT_FAILURE;
    }

    const std::string input_path = argv[1];
    const std::string output_path = argv[2];

    const int grid_x = std::stoi(argv[3]);
    const int grid_y = std::stoi(argv[4]);
    const int grid_z = std::stoi(argv[5]);

    const float iso_level = std::stof(argv[6]);
    const float percentage_extend = std::stof(argv[7]);

    // Final argument reserved for future use.
    // Keep it for a stable command-line interface.
    const float distance_ignore = std::stof(argv[8]);
    (void) distance_ignore;

    if (grid_x <= 1 || grid_y <= 1 || grid_z <= 1)
    {
        std::cerr << "Grid dimensions must be greater than 1\n";
        return EXIT_FAILURE;
    }

    if (percentage_extend < 0.0f)
    {
        std::cerr << "percentage_extend must be non-negative\n";
        return EXIT_FAILURE;
    }

    using PointNormalT = pcl::PointNormal;

    auto cloud =
        pcl::make_shared<pcl::PointCloud<PointNormalT>>();

    if (
        pcl::io::loadPLYFile<PointNormalT>(
            input_path,
            *cloud
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
        << cloud->size()
        << "\n";

    if (cloud->empty())
    {
        std::cerr << "Input cloud is empty\n";
        return EXIT_FAILURE;
    }

    std::size_t invalid_count = 0;

    for (const auto& point : cloud->points)
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
            << "Input contains invalid points or normals\n";
        return EXIT_FAILURE;
    }

    const auto& first = cloud->front();

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

    pcl::MarchingCubesHoppe<PointNormalT> marching_cubes;

    marching_cubes.setInputCloud(cloud);
    marching_cubes.setGridResolution(
        grid_x,
        grid_y,
        grid_z
    );
    marching_cubes.setIsoLevel(iso_level);
    marching_cubes.setPercentageExtendGrid(
        percentage_extend
    );

    pcl::PolygonMesh mesh;

    std::cout << "\nParameters\n";
    std::cout << "----------\n";
    std::cout
        << "Grid: "
        << grid_x << " x "
        << grid_y << " x "
        << grid_z << "\n";
    std::cout
        << "Iso level: "
        << iso_level << "\n";
    std::cout
        << "Grid extension: "
        << percentage_extend << "\n";

    const auto start =
        std::chrono::steady_clock::now();

    marching_cubes.reconstruct(mesh);

    const auto end =
        std::chrono::steady_clock::now();

    const double runtime_seconds =
        std::chrono::duration<double>(
            end - start
        ).count();

    std::cout << "\nResult\n";
    std::cout << "------\n";
    std::cout
        << "Vertices: "
        << mesh.cloud.width * mesh.cloud.height
        << "\n";
    std::cout
        << "Polygons: "
        << mesh.polygons.size()
        << "\n";
    std::cout
        << "Runtime: "
        << runtime_seconds
        << " seconds\n";

    if (mesh.polygons.empty())
    {
        std::cerr
            << "Marching Cubes generated no polygons\n";
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
