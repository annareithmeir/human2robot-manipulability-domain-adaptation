cmake_minimum_required(VERSION 3.20)
project(ma_thesis)

set(CMAKE_CXX_STANDARD 14)


find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package (Threads REQUIRED)

add_definitions(-DMAX_EXT_API_CONNECTIONS=255)

include_directories(
        include
        /usr/local/MATLAB/R2021a/extern/include/
)


add_executable(ma_thesis src/main.cpp
        src/GMM.cpp
        src/GMM_SPD.cpp
        src/Franka.cpp
        )

target_link_libraries (ma_thesis
        Eigen3::Eigen
        pthread
        dqrobotics
        dqrobotics-interface-vrep
        /usr/local/MATLAB/R2021a/extern/bin/glnxa64/libMatlabEngine.so
        /usr/local/MATLAB/R2021a/extern/bin/glnxa64/libMatlabDataArray.so
        ${CMAKE_THREAD_LIBS_INIT})


add_compile_options(-lMatlabDataArray -lMatlabEngine)
set(MATLABPATH matlab)
# addpath matlab to startup.m
set(LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:/usr/local/MATLAB/2021a/extern/bin/glnxa64:/usr/local/MATLAB/2021a/sys/os/glnxa64)
add_executable(tests
        tests/catch2.hpp
        tests/tests.cpp
        src/Franka.cpp
)

target_link_libraries (tests
        Eigen3::Eigen
        pthread
        dqrobotics
        dqrobotics-interface-vrep
        /usr/local/MATLAB/R2021a/extern/bin/glnxa64/libMatlabEngine.so
        /usr/local/MATLAB/R2021a/extern/bin/glnxa64/libMatlabDataArray.so)


