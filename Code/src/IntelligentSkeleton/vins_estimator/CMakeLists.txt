cmake_minimum_required(VERSION 3.25)
project(vins_estimator)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    geometry_msgs
    nav_msgs
    tf
    cv_bridge
    roslib
    pcl_ros
    pcl_conversions
    message_filters
    bigai_robotic_msgs
)

find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)
set(Ceres_DIR /home/shunyi/Development_Tools/ceres-build-1.14.0/install/lib/cmake/Ceres)
find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)
# find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
set(Pangolin_DIR /home/shunyi/VisualSLAM/Tools/Pangolin-0.8/install/lib/cmake/Pangolin)
find_package(Pangolin 0.8 REQUIRED)
# find_package(GLEW REQUIRED)
# find_package(OpenGL REQUIRED)

# lfc添加 PCL 查找
find_package(PCL 1.10 REQUIRED)

SET(CERES_INCLUDE_DIRS /home/shunyi/Development_Tools/ceres-build-1.14.0/install/include)
# SET()
include_directories(${catkin_INCLUDE_DIRS} 
                    ${CERES_INCLUDE_DIRS} 
                    ${EIGEN3_INCLUDE_DIR})

message("----------------------------------------------------")
message(${EIGEN3_INCLUDE_DIR})
message("----------------------------------------------------")

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


include_directories(${catkin_INCLUDE_DIRS})
include_directories(${CERES_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/sophus/)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

catkin_package()

cuda_add_library(TSDFMapper STATIC
    src/fusion/volume.cu
    src/fusion/tsdf_mapper.cpp
    src/fusion/displayer.cpp
    )

target_link_libraries(TSDFMapper
    ${Pangolin_LIBRARY} 
    # ${GLEW_LIBRARY}
    # ${OPENGL_LIBRARIES}
    ${OpenCV_LIBS} 
    )

target_include_directories(TSDFMapper
    PRIVATE src/fusion  
    ${OpenCV_INCLUDE_DIRS} 
    ${CUDA_INCLUDE_DIRS} 
    # ${GLEW_INCLUDE_DIR} 
    # ${OPENGL_INCLUDE_DIR}
    ${Pangolin_INCLUDE_DIR}
    )

# This has a cost of about 1 fps but it prohibits some hard to trace bug
target_compile_definitions(TSDFMapper PUBLIC DR_FUSION_DEBUG_SYNC_LAUNCH)

add_library(SceneGraph STATIC
    src/scene_graph/TrajTracker.cpp
    src/scene_graph/utils_control.cpp
)

target_link_libraries(SceneGraph
    ${OpenCV_LIBS}
    TSDFMapper
    )

cuda_add_library(PlaneSeg STATIC
    src/segmentation/utils.cpp
    src/segmentation/SurfaceExtraction.cu
)

target_link_libraries(PlaneSeg
    ${OpenCV_LIBS} 
)

target_include_directories(PlaneSeg
    PRIVATE src/segmentation  
    ${OpenCV_INCLUDE_DIRS} 
)

add_executable(vins_estimator
    src/estimator_node.cpp
    src/parameters.cpp
    src/estimator.cpp
    src/feature_manager.cpp
    src/factor/pose_local_parameterization.cpp
    src/factor/projection_factor.cpp
    src/factor/projection_td_factor.cpp
    src/factor/marginalization_factor.cpp
    src/factor/depth_factor.cpp
    src/factor/projection_xyz_factor.cpp
    src/factor/vbg_factor.cpp
    src/factor/surface_factor.cpp
    src/utility/utility.cpp
    src/utility/visualization.cpp
    src/utility/CameraPoseVisualization.cpp
    src/initial/solve_5pts.cpp
    src/initial/initial_aligment.cpp
    src/initial/initial_sfm.cpp
    src/initial/initial_ex_rotation.cpp
    )


target_link_libraries(vins_estimator 
    ${catkin_LIBRARIES} 
    ${OpenCV_LIBS} 
    ${CERES_LIBRARIES} 
    TSDFMapper 
    SceneGraph 
    PlaneSeg
    CUDA::cublas
    CUDA::cudart
    CUDA::cusparse
    pcl_visualization   # <-- lfc添加这一行
    )

#     #新加的 lfc 1
# add_executable(subsvm
#     src/subsvm.cpp
#     )
#     #新加的 lfc 1
# target_link_libraries(subsvm
#     ${catkin_LIBRARIES}
#     ${CMAKE_THREAD_LIBS_INIT}
#     )


# add_executable(scene_graph_node
#     src/scene_graph_node.cpp)


# target_link_libraries(scene_graph_node 
#     ${catkin_LIBRARIES}
#     SceneGraph)

