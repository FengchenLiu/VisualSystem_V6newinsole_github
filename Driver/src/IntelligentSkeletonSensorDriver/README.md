# ROS Driver for BIGAI Intelligent Skeleton
These are packages for using Intel RealSense cameras D455 (with IMU) and PKU skeleton sensors with ROS.

This version supports ROS-Melodic and ROS-Noetic distributions.

LibRealSense supported version: v2.44.0 (see [realsense2_camera release notes](https://github.com/IntelRealSense/realsense-ros/releases))

## Installation Instructions

### Ubuntu
   - #### Install [ROS Melodic](http://wiki.ros.org/melodic/Installation/Ubuntu) on Ubuntu 18.04 or [ROS Noetic](http://wiki.ros.org/noetic/Installation/Ubuntu) on Ubuntu 20.04.


### Install Intel Realsense Driver (librealsense)

   ### Step 1: Install the latest Intel&reg; RealSense&trade; SDK 2.0
   - Build from sources by downloading the latest [Intel&reg; RealSense&trade; SDK 2.0](https://github.com/IntelRealSense/librealsense/releases/tag/v2.44.0) and follow the instructions under [Linux Installation](https://github.com/IntelRealSense/librealsense/blob/master/doc/installation.md)


   ### Step 2: Build this Repo; ROS from Sources
   - Create a [catkin](http://wiki.ros.org/catkin#Installing_catkin) workspace
   *Ubuntu*
   ```bash
   mkdir -p ~/catkin_ws/src
   cd ~/catkin_ws/src/
   ```

   - Clone this repo into 'catkin_ws/src/'
   ```bash
   git clone https://github.com/zhyhit/IntelligentSkeletonSensorDriver.git
   cd ..
   ```
   - Make sure all dependent packages are installed. You can check .travis.yml file for reference.
   - Specifically, make sure that the ros package *ddynamic_reconfigure* is installed. 
   ```bash
   sudo apt-get install ros-melodic-ddynamic-reconfigure
   ```

   - Then, you can build the ROS node. 
   ```bash
   catkin_make .. -DCMAKE_BUILD_TYPE=Release
   ```

## Usage Instructions

### Start the camera node
To start the camera node in ROS:

```bash
roslaunch realsense2_camera bigai.launch
```

This will stream all sensors and publish on the appropriate ROS topics.

### Published Topics
The published topics differ according to the device and parameters.
After running the above command with D455 and other sensors attached, the following list of topics will be available (This is a partial list. For full one type `rostopic list`):
- /camera/aligned_depth_to_color/image_raw
- /camera/color/camera_info
- /camera/color/image_raw
- /camera/imu
- /clock
- /press
- /rosout
- /rosout_agg
- /skelenton_imu_1
- /skelenton_imu_2
- /skelenton_imu_3
- /skelenton_imu_5
- /skelenton_imu_8

These topics should be recorded in a ROS bag if you want to save a sequence for experiment.
```bash
rosbag record /camera/aligned_depth_to_color/image_raw /camera/color/camera_info /camera/color/image_raw  /camera/imu  /press /skelenton_imu_1 /skelenton_imu_2 /skelenton_imu_3 /skelenton_imu_5 /skelenton_imu_8  -o bigai_all_sensor.bag
```

To check the bag, you can use the command
```bash
rosbag info YOUR_BAG.bag
```

## License
Copyright 2018 Intel Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this project except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

**Other names and brands may be claimed as the property of others*
