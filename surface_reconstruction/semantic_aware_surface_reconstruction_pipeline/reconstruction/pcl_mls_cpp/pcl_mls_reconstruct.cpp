#include <pcl/common/centroid.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/surface/mls.h>

#include <cmath>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

struct Options {
  std::filesystem::path input;
  std::filesystem::path output;
  double search_radius = 0.1;
  double voxel_size = 0.0;
  int polynomial_order = 2;
  int threads = 0;
  std::string upsampling = "none";
  double upsampling_radius = 0.0;
  double upsampling_step = 0.0;
};

[[noreturn]] void usage(const std::string& message = {}) {
  if (!message.empty()) {
    std::cerr << "ERROR: " << message << "\n\n";
  }
  std::cerr
      << "Usage: pcl_mls_reconstruct --input in.pcd --output out.pcd\\\n\n"
      << "  --search-radius M --polynomial-order N [--voxel-size M]\\\n\n"
      << "  [--threads N] [--upsampling none|sample_local_plane]\\\n\n"
      << "  [--upsampling-radius M --upsampling-step M]\n";
  std::exit(message.empty() ? 0 : 2);
}

template <typename T>
T parse_number(const std::string& text, const std::string& flag);

template <>
double parse_number<double>(const std::string& text, const std::string& flag) {
  try {
    std::size_t used = 0;
    const double value = std::stod(text, &used);
    if (used != text.size() || !std::isfinite(value)) {
      throw std::invalid_argument("invalid");
    }
    return value;
  } catch (const std::exception&) {
    usage("Invalid value for " + flag + ": " + text);
  }
}

template <>
int parse_number<int>(const std::string& text, const std::string& flag) {
  try {
    std::size_t used = 0;
    const int value = std::stoi(text, &used);
    if (used != text.size()) {
      throw std::invalid_argument("invalid");
    }
    return value;
  } catch (const std::exception&) {
    usage("Invalid value for " + flag + ": " + text);
  }
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string flag = argv[index];
    if (flag == "--help" || flag == "-h") {
      usage();
    }
    if (index + 1 >= argc) {
      usage("Missing value after " + flag);
    }
    const std::string value = argv[++index];
    if (flag == "--input") {
      options.input = value;
    } else if (flag == "--output") {
      options.output = value;
    } else if (flag == "--search-radius") {
      options.search_radius = parse_number<double>(value, flag);
    } else if (flag == "--voxel-size") {
      options.voxel_size = parse_number<double>(value, flag);
    } else if (flag == "--polynomial-order") {
      options.polynomial_order = parse_number<int>(value, flag);
    } else if (flag == "--threads") {
      options.threads = parse_number<int>(value, flag);
    } else if (flag == "--upsampling") {
      options.upsampling = value;
    } else if (flag == "--upsampling-radius") {
      options.upsampling_radius = parse_number<double>(value, flag);
    } else if (flag == "--upsampling-step") {
      options.upsampling_step = parse_number<double>(value, flag);
    } else {
      usage("Unknown option: " + flag);
    }
  }

  if (options.input.empty() || options.output.empty()) {
    usage("--input and --output are required");
  }
  if (!(options.search_radius > 0.0)) {
    usage("--search-radius must be positive");
  }
  if (options.voxel_size < 0.0 || options.polynomial_order < 1 ||
      options.threads < 0) {
    usage("voxel size and thread count must be non-negative; order >= 1");
  }
  if (options.upsampling != "none" &&
      options.upsampling != "sample_local_plane") {
    usage("--upsampling must be none or sample_local_plane");
  }
  if (options.upsampling == "sample_local_plane" &&
      (!(options.upsampling_radius > 0.0) ||
       !(options.upsampling_step > 0.0))) {
    usage("sample_local_plane requires positive radius and step");
  }
  return options;
}

bool finite_point_normal(const pcl::PointNormal& point) {
  return std::isfinite(point.x) && std::isfinite(point.y) &&
         std::isfinite(point.z) && std::isfinite(point.normal_x) &&
         std::isfinite(point.normal_y) && std::isfinite(point.normal_z);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);

    auto input = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(options.input.string(), *input) < 0) {
      throw std::runtime_error("Could not read " + options.input.string());
    }
    if (input->empty()) {
      throw std::runtime_error("Input cloud is empty");
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered = input;
    if (options.voxel_size > 0.0) {
      filtered = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
      pcl::VoxelGrid<pcl::PointXYZ> voxel;
      voxel.setInputCloud(input);
      const float leaf = static_cast<float>(options.voxel_size);
      voxel.setLeafSize(leaf, leaf, leaf);
      voxel.filter(*filtered);
    }
    if (filtered->empty()) {
      throw std::runtime_error("Voxel filtering removed every point");
    }

    // Fit each tile/group in a local coordinate system. This avoids loss of
    // numerical precision when Waymo world coordinates are large.
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*filtered, centroid);
    for (auto& point : filtered->points) {
      point.x -= centroid[0];
      point.y -= centroid[1];
      point.z -= centroid[2];
    }

    auto tree = pcl::make_shared<pcl::search::KdTree<pcl::PointXYZ>>();
    pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointNormal> mls;
    mls.setInputCloud(filtered);
    mls.setSearchMethod(tree);
    mls.setComputeNormals(true);
    mls.setSearchRadius(options.search_radius);
    mls.setPolynomialOrder(options.polynomial_order);
    if (options.threads > 0) {
      mls.setNumberOfThreads(static_cast<unsigned int>(options.threads));
    }
    if (options.upsampling == "sample_local_plane") {
      mls.setUpsamplingMethod(
          pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointNormal>::SAMPLE_LOCAL_PLANE);
      mls.setUpsamplingRadius(options.upsampling_radius);
      mls.setUpsamplingStepSize(options.upsampling_step);
      // Required in the PCL build used for the successful ground experiment:
      // without cached MLS fits, SAMPLE_LOCAL_PLANE can behave like projection-only.
      mls.setCacheMLSResults(true);
    } else {
      mls.setUpsamplingMethod(
          pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointNormal>::NONE);
    }

    auto raw_output = pcl::make_shared<pcl::PointCloud<pcl::PointNormal>>();
    mls.process(*raw_output);

    pcl::PointCloud<pcl::PointNormal> output;
    output.reserve(raw_output->size());
    for (const auto& point : raw_output->points) {
      if (!finite_point_normal(point)) {
        continue;
      }
      const double norm = std::sqrt(
          static_cast<double>(point.normal_x) * point.normal_x +
          static_cast<double>(point.normal_y) * point.normal_y +
          static_cast<double>(point.normal_z) * point.normal_z);
      if (norm > 1e-8) {
        output.push_back(point);
      }
    }
    // Restore Waymo-world translation after local MLS fitting.
    for (auto& point : output.points) {
      point.x += centroid[0];
      point.y += centroid[1];
      point.z += centroid[2];
    }
    output.width = static_cast<std::uint32_t>(output.size());
    output.height = 1;
    output.is_dense = true;
    if (output.empty()) {
      throw std::runtime_error("MLS produced no finite point normals");
    }

    std::filesystem::create_directories(options.output.parent_path());
    if (pcl::io::savePCDFileBinary(options.output.string(), output) < 0) {
      throw std::runtime_error("Could not write " + options.output.string());
    }

    std::cout << "PCL MLS: " << input->size() << " input, "
              << filtered->size() << " after voxelization, " << output.size()
              << " output point normals\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "ERROR: " << error.what() << '\n';
    return 1;
  }
}