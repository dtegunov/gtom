CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(gtom)

FIND_PACKAGE(OpenMP REQUIRED)
FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(Threads REQUIRED)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

#GTOM

INCLUDE(FindCUDA)

INCLUDE_DIRECTORIES(/usr/local/cuda/include include)

LINK_DIRECTORIES(/usr/local/cuda/lib64)

FILE(GLOB_RECURSE SOURCES "src/*.cu")
LIST(APPEND SOURCES src/Helpers/KaiserBessel.cpp)
LIST(APPEND SOURCES src/IO/em.cpp src/IO/mrc.cpp src/IO/raw.cpp)
LIST(APPEND SOURCES src/Optimization/dlib/all/source.cpp)
LIST(APPEND SOURCES src/Optimization/polynomials/Polynomial.cpp src/Optimization/polynomials/PolynomialRootFinder.cpp)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1y -w")
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pthread")

SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-w")
SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-gencode arch=compute_52,code=sm_52")
SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-restrict")
SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-lineinfo")
SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-std=c++11")
SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-cudart=static")
#SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};--ptxas-options=-v")

CUDA_ADD_LIBRARY(gtom STATIC ${SOURCES})

TARGET_LINK_LIBRARIES(gtom libcudart.so libcufft.so libcublas.so libcurand.so ${CMAKE_THREAD_LIBS_INIT})


#Google Test

INCLUDE_DIRECTORIES(GoogleTest GoogleTest/include)
SET(SOURCES "")
LIST(APPEND SOURCES GoogleTest/src/gtest-all.cc GoogleTest/src/gtest_main.cc)
ADD_LIBRARY(googletest STATIC ${SOURCES})
TARGET_LINK_LIBRARIES(googletest ${CMAKE_THREAD_LIBS_INIT})


#GTOM Test

INCLUDE_DIRECTORIES(GoogleTest/include /usr/local/cuda/include)
FILE(GLOB_RECURSE SOURCES "GTOMTest/*.cpp")
ADD_EXECUTABLE(gtomtest ${SOURCES})
TARGET_LINK_LIBRARIES(gtomtest gtom googletest)
