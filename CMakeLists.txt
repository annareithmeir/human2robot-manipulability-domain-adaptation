cmake_minimum_required(VERSION 3.20)
project(ma_thesis)

set(CMAKE_CXX_STANDARD 14)


find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package (Threads REQUIRED)
find_package(PythonLibs REQUIRED)

add_definitions(-DMAX_EXT_API_CONNECTIONS=255)

add_compile_options(-lMatlabDataArray -lMatlabEngine)
set(MATLABPATH matlab matlab/extra matlab/@rHuManModel)  # "addpath matlab ..." to startup.m must be done additonally!
set(LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:/usr/local/MATLAB/2021a/extern/bin/glnxa64:/usr/local/MATLAB/2021a/sys/os/glnxa64)

include_directories(
        include
        /usr/local/MATLAB/R2021a/extern/include/
        ${PYTHON_INCLUDE_DIRS}
)

add_library(gmm src/GMM_SPD.cpp src/GMM.cpp)
target_link_libraries(gmm Eigen3::Eigen)

add_library(franka src/Franka.cpp)
target_link_libraries(franka
        pthread
        Eigen3::Eigen
        ${PYTHON_LIBRARIES}
        dqrobotics
        dqrobotics-interface-vrep
#        /usr/local/MATLAB/R2021a/extern/bin/glnxa64/libMatlabEngine.so
#        /usr/local/MATLAB/R2021a/extern/bin/glnxa64/libMatlabDataArray.so
        ${CMAKE_THREAD_LIBS_INIT})

add_library(controller src/control.cpp)
target_link_libraries(controller
        pthread
        Eigen3::Eigen
        ${PYTHON_LIBRARIES}
        dqrobotics
        dqrobotics-interface-vrep
        /usr/local/MATLAB/R2021a/extern/bin/glnxa64/libMatlabEngine.so
        /usr/local/MATLAB/R2021a/extern/bin/glnxa64/libMatlabDataArray.so)

add_library(learn src/learn.cpp)
target_link_libraries(learn
        ${PYTHON_LIBRARIES}
        pthread
        Eigen3::Eigen
        gmm)


add_executable(ma_thesis src/main.cpp
        )

target_link_libraries (ma_thesis
        Eigen3::Eigen
        ${PYTHON_LIBRARIES}
        gmm
        franka
        controller
        learn)


add_executable(calibration_process src/calibrationProcess.cpp
        )

target_link_libraries (calibration_process
        Eigen3::Eigen
        ${PYTHON_LIBRARIES}
        gmm
        franka
        controller
        learn)


add_executable(tests
        tests/catch2.hpp
        tests/tests.cpp
)

target_link_libraries (tests
        Eigen3::Eigen
        ${PYTHON_LIBRARIES}
        dqrobotics
        dqrobotics-interface-vrep
        /usr/local/MATLAB/R2021a/extern/bin/glnxa64/libMatlabEngine.so
        /usr/local/MATLAB/R2021a/extern/bin/glnxa64/libMatlabDataArray.so
        gmm
        franka
        controller
        learn)


