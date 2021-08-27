cmake_minimum_required(VERSION 3.20)
project(ma_thesis)

set(CMAKE_CXX_STANDARD 14)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
include_directories(include)

add_executable(ma_thesis tests/main.cpp src/SPD.cpp include/SPD.h src/GMM.cpp include/GMM.h include/utils.h)
target_link_libraries (ma_thesis Eigen3::Eigen)
