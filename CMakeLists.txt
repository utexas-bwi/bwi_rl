cmake_minimum_required(VERSION 2.8.3)
project(bwi_rl)

find_package(catkin REQUIRED COMPONENTS bwi_tools)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS bwi_tools
)
