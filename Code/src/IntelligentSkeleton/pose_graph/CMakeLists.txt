cmake_minimum_required(VERSION 3.25)
project(pose_graph)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
#-DEIGEN_USE_MKL_ALL")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    nav_msgs
    camera_model
    cv_bridge
    roslib
    pcl_ros
    pcl_conversions
    message_filters
    )

find_package(OpenCV)
set(Ceres_DIR /home/shunyi/Development_Tools/ceres-build-1.14.0/install/lib/cmake/Ceres)
find_package(Ceres REQUIRED)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)

SET(CERES_INCLUDE_DIRS /home/shunyi/Development_Tools/ceres-build-1.14.0/install/include)
include_directories(${catkin_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})

catkin_package()

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--std=c++14;-O3)

# See: http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=sm_60)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode=arch=compute_60,code=sm_60)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode=arch=compute_61,code=sm_61)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode=arch=compute_70,code=sm_70)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode=arch=compute_75,code=sm_75)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode=arch=compute_75,code=compute_75)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-lineinfo)

cuda_add_library(tsdf_fusion STATIC
    src/fusion/volume.cu
    src/fusion/tsdf_mapper.cu
    )

target_include_directories(tsdf_fusion
    PRIVATE src/fusion  ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS}
    )

# This has a cost of about 1 fps but it prohibits some hard to trace bug
target_compile_definitions(tsdf_fusion PUBLIC DR_FUSION_DEBUG_SYNC_LAUNCH)

add_executable(pose_graph
    src/pose_graph_node.cpp
    src/pose_graph.cpp
    src/keyframe.cpp
    src/utility/CameraPoseVisualization.cpp
    src/loop_closure/DBoW/BowVector.cpp
    src/loop_closure/DBoW/FBrief.cpp
    src/loop_closure/DBoW/FeatureVector.cpp
    src/loop_closure/DBoW/QueryResults.cpp
    src/loop_closure/DBoW/ScoringObject.cpp
    src/loop_closure/DUtils/Random.cpp
    src/loop_closure/DUtils/Timestamp.cpp
    src/loop_closure/DVision/BRIEF.cpp
    src/loop_closure/VocabularyBinary.cpp
    )

target_link_libraries(pose_graph ${catkin_LIBRARIES}  ${OpenCV_LIBS} ${CERES_LIBRARIES} tsdf_fusion)
message("catkin_lib  ${catkin_LIBRARIES}")
