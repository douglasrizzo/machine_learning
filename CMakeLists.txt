cmake_minimum_required(VERSION 3.8)
project(machine_learning)

set(CMAKE_CXX_STANDARD 11)

# this OpenMP hack was found here https://stackoverflow.com/a/12404666
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(SOURCE_FILES main.cpp include/KNN.hpp include/LeastSquares.hpp include/Matrix.hpp include/PCA.hpp)
add_executable(machine_learning ${SOURCE_FILES})