cmake_minimum_required(VERSION 3.0.2)
project(chessboard_detector)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  cv_bridge
  tf2_ros
  tf
)

catkin_package(
  CATKIN_DEPENDS
    roscpp
    rospy
    std_msgs
    sensor_msgs
    geometry_msgs
    cv_bridge
    tf2_ros
    tf
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

# 添加Python脚本安装
catkin_install_python(PROGRAMS
  scripts/chessboard_detector.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# 安装launch文件和config文件
install(DIRECTORY
  launch/
  config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

