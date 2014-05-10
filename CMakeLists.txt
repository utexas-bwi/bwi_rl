cmake_minimum_required(VERSION 2.8.3)
project(bwi_rl)

find_package(catkin REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME} ${PROJECT_NAME}_json
)

## Build
ADD_DEFINITIONS(-std=c++0x)

include_directories(
  include
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


## Test Binaries
add_executable(test_mt_mcts
  test/test_mt_mcts.cpp
)
