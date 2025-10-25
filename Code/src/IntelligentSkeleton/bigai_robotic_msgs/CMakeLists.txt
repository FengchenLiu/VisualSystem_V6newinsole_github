project(bigai_robotic_msgs)
cmake_minimum_required(VERSION 3.25)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR})

find_package(catkin REQUIRED COMPONENTS
  std_msgs
  nav_msgs
  geometry_msgs
  actionlib
  message_generation
)

add_message_files(
  DIRECTORY msg
  FILES
  PlaneSeg.msg
  PlaneSegVec.msg
  Press.msg
)


generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  nav_msgs
)

catkin_package(
  CATKIN_DEPENDS
  message_runtime
)
