cmake_minimum_required(VERSION 3.16)

project(waymo_pcl_reconstruction)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(
    PCL REQUIRED COMPONENTS
    common
    io
    search
    kdtree
    features
    surface
)

add_executable(
    pcl_reconstruct
    pcl_reconstruct.cpp
)

target_include_directories(
    pcl_reconstruct
    PRIVATE
    ${PCL_INCLUDE_DIRS}
)

target_link_directories(
    pcl_reconstruct
    PRIVATE
    ${PCL_LIBRARY_DIRS}
)

target_compile_definitions(
    pcl_reconstruct
    PRIVATE
    ${PCL_DEFINITIONS}
)

target_link_libraries(
    pcl_reconstruct
    PRIVATE
    ${PCL_LIBRARIES}
)