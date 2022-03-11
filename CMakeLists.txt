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

INCLUDE_DIRECTORIES(${CUDA_HOME}/include include)

LINK_DIRECTORIES(${CUDA_HOME}/lib64)

FILE(GLOB_RECURSE SOURCES "src/*.cu")
LIST(APPEND SOURCES src/Helpers/KaiserBessel.cpp)
LIST(APPEND SOURCES src/Helpers/Misc.cpp)
LIST(APPEND SOURCES src/IO/em.cpp src/IO/mrc.cpp src/IO/raw.cpp)
LIST(APPEND SOURCES src/Optimization/dlib/all/source.cpp)
LIST(APPEND SOURCES src/Optimization/polynomials/Polynomial.cpp src/Optimization/polynomials/PolynomialRootFinder.cpp)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -std=c++11 -mavx -w")
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pthread")

SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-w")
SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-gencode arch=compute_52,code=sm_52")
SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-gencode arch=compute_61,code=sm_61")
SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-gencode arch=compute_75,code=sm_75")
SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-gencode arch=compute_86,code=sm_86")
SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-restrict")
SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-use_fast_math")
SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-lineinfo")
SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-std=c++11")
SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-cudart=static")
#SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};--ptxas-options=-v")

CUDA_ADD_LIBRARY(gtom STATIC ${SOURCES})

TARGET_LINK_LIBRARIES(gtom libcudart.so libcufft.so libcublas.so libcurand.so ${CMAKE_THREAD_LIBS_INIT})

