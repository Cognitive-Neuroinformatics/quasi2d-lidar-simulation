cmake_minimum_required(VERSION 3.10)

project(gaussian_ply_to_mesh LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(
    PCL REQUIRED COMPONENTS
    common
    io
    features
    search
    kdtree
    surface
)

add_executable(
    gaussian_ply_to_mesh
    gaussian_ply_to_mesh.cpp
)

target_include_directories(
    gaussian_ply_to_mesh
    PRIVATE
    ${PCL_INCLUDE_DIRS}
)

target_link_directories(
    gaussian_ply_to_mesh
    PRIVATE
    ${PCL_LIBRARY_DIRS}
)

target_compile_definitions(
    gaussian_ply_to_mesh
    PRIVATE
    ${PCL_DEFINITIONS}
)

target_link_libraries(
    gaussian_ply_to_mesh
    PRIVATE
    ${PCL_LIBRARIES}
)