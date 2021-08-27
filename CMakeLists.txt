cmake_minimum_required(VERSION 3.20)
project(ma_thesis)

set(CMAKE_CXX_STANDARD 14)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(ma_thesis main.cpp)
target_link_libraries (ma_thesis Eigen3::Eigen)
