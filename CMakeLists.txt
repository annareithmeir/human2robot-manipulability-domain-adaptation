cmake_minimum_required(VERSION 3.20)
project(ma_thesis)

set(CMAKE_CXX_STANDARD 14)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
#find_package(Catch2 3 REQUIRED)
include_directories(include)

add_executable(ma_thesis src/main.cpp src/GMM.cpp src/GMM_SPD.cpp include/GMM.h include/GMM_SPD.h include/utils.h)
target_link_libraries (ma_thesis Eigen3::Eigen)

#add_executable(tests tests/tests.cpp)
#target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)
