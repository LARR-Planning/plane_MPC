cmake_minimum_required(VERSION 2.8.3)
project(optim_traj_gen)

add_compile_options(-std=c++11)
set(CMAKE_BUILD_TYPE Release)
## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS 
    roscpp  
    geometry_msgs
    nav_msgs
    voxblox_ros
)


# additional dependencies 

set(qpOASES_SRC /home/jbs/lib/qpOASES-3.2.1)
file(GLOB_RECURSE qpOASES_LIBS ${qpOASES_SRC}/src/*.cpp)

find_package(dynamicEDT3D REQUIRED)
include_directories(${DYNAMICEDT3D_INCLUDE_DIRS})
link_libraries(${DYNAMICEDT3D_LIBRARIES})

find_package(octomap REQUIRED)
include_directories(${OCTOMAP_INCLUDE_DIRS})
link_libraries(${OCTOMAP_LIBRARIES})

find_package(Eigen3 REQUIRED NO_MODULE)


catkin_package(
  INCLUDE_DIRS include
  LIBRARIES optim_traj_gen
)


include_directories(
include
${catkin_INCLUDE_DIRS}
)

# module1 : standalone node for test 
add_executable(chomp_test_app
    src/chomp_main_test.cpp 
    src/chomp_utils.cpp
    src/chomp_subroutine.cpp
    src/chomp_ros_wrapper.cpp
    ${qpOASES_LIBS}
)

target_link_libraries(chomp_test_app
  ${catkin_LIBRARIES}
)





