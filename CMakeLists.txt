cmake_minimum_required(VERSION 3.20)
project(ma_thesis)

set(CMAKE_CXX_STANDARD 14)
#set( catkin_DIR /opt/ros/melodic/share/catkin/cmake )
#set( CMAKE_PREFIX_PATH /opt/ros/melodic )

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

#find_package(Catch2 3 REQUIRED)
#find_package(catkin REQUIRED COMPONENTS
#        eigen_conversions
#        pcl_conversions
#        pcl_ros
#        roscpp
#        sensor_msgs
#        std_msgs
#        )
find_package (Threads REQUIRED)

#add_definitions(-DIL_STD)
#add_definitions(-DNON_MATLAB_PARSING)
add_definitions(-DMAX_EXT_API_CONNECTIONS=255)
#add_definitions(-DDO_NOT_USE_SHARED_MEMORY)

#catkin_package(
#        INCLUDE_DIRS include
#)

include_directories(
        include
#        ${catkin_INCLUDE_DIRS}
)

#set(Catch2_DIR /home/nnrthmr/Catch2/CMake)
add_executable(ma_thesis src/main.cpp src/GMM.cpp src/GMM_SPD.cpp include/GMM.h include/GMM_SPD.h include/utils.h src/Franka.cpp include/Franka.h include/SPD_Utils.h)
target_link_libraries (ma_thesis
        Eigen3::Eigen
        pthread
        dqrobotics
        dqrobotics-interface-vrep)


#find_package(Catch2 REQUIRED)
add_executable(tests tests/catch2.hpp tests/tests.cpp src/Franka.cpp include/Franka.h)
target_link_libraries (tests
        Eigen3::Eigen
        pthread
        dqrobotics
        dqrobotics-interface-vrep)
#target_link_libraries(functionaltests PRIVATE Catch2::Catch2WithMain)
