cmake_minimum_required(VERSION 2.8.3)
project(bwi_rl)

find_package(catkin REQUIRED COMPONENTS cmake_modules)
find_package(TBB REQUIRED)

catkin_package(
  INCLUDE_DIRS include ${TBB_INCLUDE_DIRS}
  LIBRARIES ${PROJECT_NAME} ${PROJECT_NAME}_json ${TBB_LIBRARIES}
)

## Build
ADD_DEFINITIONS(-std=c++0x)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

## The main bwi_rl library
add_library(${PROJECT_NAME}_json
  src/json/json_reader.cpp
  src/json/json_value.cpp
  src/json/json_writer.cpp
)

add_library(${PROJECT_NAME}
  src/common/Enum.cpp
  src/common/Point2D.cpp
  src/common/tinymt32.cpp
  src/common/Util.cpp
)
