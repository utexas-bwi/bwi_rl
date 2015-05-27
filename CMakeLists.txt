cmake_minimum_required(VERSION 2.8.3)
project(bwi_rl)

find_package(catkin REQUIRED COMPONENTS bwi_tools pluginlib)
find_package(Boost REQUIRED COMPONENTS program_options serialization)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS bwi_tools pluginlib
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)

add_executable(evaluator
  src/evaluator.cpp
)
target_link_libraries(evaluator
  ${catkin_LIBRARIES}
  ${Boost_LIBRARIES}
)

add_executable(vi_example
  examples/vi/vi_example.cpp
)
target_link_libraries(vi_example
  ${catkin_LIBRARIES}
  ${Boost_LIBRARIES}
)

## Mark executables and/or libraries for installation
install(TARGETS
  evaluator
  vi_example
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
