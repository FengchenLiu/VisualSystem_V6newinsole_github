# RGBD-Inertial State Estimator and TSDF Surface Mapping and Segmentation for BIGAI Intelligent Skeleton
Based one open source SLAM framework [VINS-Mono](https://github.com/HKUST-Aerial-Robotics/VINS-Mono) 
and its extended version [VINS-RGBD](https://github.com/STAR-Center/VINS-RGBD).

The approach contains
+ Depth-enhanced visual-inertial initialization process.
+ Depth-enhanced Visual-inertial odometry by adding depth residuals.
+ Real-time TSDF mapping with RGB images, depth images and poses (TSDF Volume with certain size follows the device)
+ Depth-based surface segmentation, surface association and global surface reconstruction
+ (TO DO) Scene graph building from surface reconstruction result.

## 1. Prerequisites
1.1. **Ubuntu** 20.04.

1.2. **ROS** version Noetic fully installation

1.3. **Ceres Solver**
Follow [Ceres Installation](http://ceres-solver.org/installation.html)

1.4. **Pangolin**
Follow [Pangolin Installation](https://github.com/stevenlovegrove/Pangolin)

1.5 **Cuda**
Version >= 6.0 is required and the memory of your GPU should be larger than 4GB (maybe 2GB is ok)

## 2. Build
```bash
mkdir catkin_ws && cd catkin_ws
git clone https://github.com/zhyhit/IntelligentSkeleton.git src
catkin_make
```

## 3. Run with ROS bag
Launch a terminal in the path of workspace
```bash
source devel/setup.bash
roslaunch vins_estimator bigai_rgbd.launch
```
Launch another terminal to play the bag
```bash
rosbag play XXX.bag
```


## 4. Run the SceneGraph Test Node
Launch a terminal in the path of workspace
```bash
source devel/setup.bash
roslaunch vins_estimator scene_graph.launch
```

## 4. Datasets (TODO)
Recording by RealSense D435i. Contain 9 bags in three different applicaions:
+ [Handheld](https://star-center.shanghaitech.edu.cn/seafile/d/0ea45d1878914077ade5/)

Topics:
+ depth topic: /camera/aligned_depth_to_color/image_raw
+ color topic: /camera/color/image_raw
+ imu topic: /camera/imu
+ camera info topic: /camera/color/camera_info

## 5. Licence
The source code is released under [GPLv3](http://www.gnu.org/licenses/) license.