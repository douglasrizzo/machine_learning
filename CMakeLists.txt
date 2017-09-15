cmake_minimum_required(VERSION 3.8)
project(machine_learning)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp include/KNN.hpp)
add_executable(machine_learning ${SOURCE_FILES})