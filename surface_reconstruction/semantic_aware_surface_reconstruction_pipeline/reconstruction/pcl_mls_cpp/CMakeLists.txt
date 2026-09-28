cmake_minimum_required(VERSION 3.16)
project(pcl_mls_reconstruct LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(PCL REQUIRED COMPONENTS common io filters kdtree search surface)

add_executable(pcl_mls_reconstruct pcl_mls_reconstruct.cpp)
target_include_directories(pcl_mls_reconstruct PRIVATE ${PCL_INCLUDE_DIRS})
target_link_directories(pcl_mls_reconstruct PRIVATE ${PCL_LIBRARY_DIRS})
target_compile_definitions(pcl_mls_reconstruct PRIVATE ${PCL_DEFINITIONS})
target_link_libraries(pcl_mls_reconstruct PRIVATE ${PCL_LIBRARIES})

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(pcl_mls_reconstruct PRIVATE -Wall -Wextra -Wpedantic)
endif()