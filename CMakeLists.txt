cmake_minimum_required(VERSION 3.8)
project(machine_learning)

set(CMAKE_CXX_STANDARD 11)

# this OpenMP hack was found here https://stackoverflow.com/a/12404666
find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()

SET(GCC_ERROR_RETURN_TYPE "-Werror=return-type")
add_definitions(${GCC_ERROR_RETURN_TYPE})

#SET(GCC_COVERAGE_COMPILE_FLAGS "-fprofile-arcs -ftest-coverage")
#SET(GCC_COVERAGE_LINK_FLAGS    "-lgcov")
#
#add_definitions(${GCC_COVERAGE_COMPILE_FLAGS})
#add_definitions(${GCC_COVERAGE_LINK_FLAGS})

# this is necessary for debugging in CLion
SET(CMAKE_BUILD_TYPE Debug)

set(SOURCE_FILES main.cpp include/KNN.hpp include/LeastSquares.hpp include/matrix/Matrix.hpp include/PCA.hpp include/LDA.hpp include/KMeans.hpp include/Metrics.hpp include/MLP.hpp include/ClassifierUtils.hpp include/NaiveBayes.hpp include/GridWorld.hpp include/Timer.hpp)
add_executable(machine_learning ${SOURCE_FILES})
