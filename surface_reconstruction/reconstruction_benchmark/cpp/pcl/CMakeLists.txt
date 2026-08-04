cmake_minimum_required(VERSION 3.16)

project(gp3_reconstruction)

find_package(PCL REQUIRED COMPONENTS
    common
    io
    features
    kdtree
    search
    surface
)

add_executable(
    gp3_reconstruction
    gp3_reconstruction.cpp
)

target_include_directories(
    gp3_reconstruction
    PRIVATE
    ${PCL_INCLUDE_DIRS}
)

target_link_libraries(
    gp3_reconstruction
    PRIVATE
    ${PCL_LIBRARIES}
)

target_compile_definitions(
    gp3_reconstruction
    PRIVATE
    ${PCL_DEFINITIONS}
)

set_target_properties(
    gp3_reconstruction
    PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
