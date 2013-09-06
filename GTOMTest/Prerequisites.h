#pragma once


#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <fstream>

#include <cufft.h>
#include <cublas_v2.h>

#include "gtest\gtest.h"
#include "..\Functions.cuh"

using namespace std;

/**
 * \brief Asserts that |expected-actual| is below the specified margin
 * \param[in] expected	Expected value
 * \param[in] actual	Actual value
 * \param[in] range		Maximum margin of error that passes the test
 */
#define ASSERT_ABSOLUTE_RANGE(expected, actual, range) \
		GTEST_CHECK_(abs(actual - expected) <= range) \
		<< "\nExpected: " << expected << ", Actual: " << actual \
		<< ", Difference: " << abs(expected - actual)

/**
 * \brief Asserts that |(expected-actual)/expected| is below the specified margin
 * \param[in] expected	Expected value
 * \param[in] actual	Actual value
 * \param[in] range		Maximum margin of error that passes the test
 */
#define ASSERT_RELATIVE_RANGE(expected, actual, range) \
		if (expected == 0.0f && actual != 0.0f) \
		{ \
			GTEST_CHECK_(false) << "\nExpected: " << expected << ", Actual: " << actual; \
		} \
		else if (expected != 0.0f && actual != 0.0f) \
		{ \
			GTEST_CHECK_(abs((actual - expected) / expected) <= range) \
			<< "\nExpected: " << expected << ", Actual: " << actual \
			<< ", Relative difference: " << (abs(expected - actual) / expected); \
		}

/**
 * \brief Asserts that |expected-actual| is below the specified margin for every value pair
 * \param[in] expected	Array with expected values
 * \param[in] actual	Array with actual values
 * \param[in] n			Array element count
 * \param[in] range		Maximum margin of error that passes the test
 */
void ASSERT_ARRAY_ABSOLUTE_RANGE(float* expected, float* actual, int n, float range);

/**
 * \brief Asserts that |(expected-actual)/expected| is below the specified margin for every value pair
 * \param[in] expected	Array with expected values
 * \param[in] actual	Array with actual values
 * \param[in] n			Array element count
 * \param[in] range		Maximum margin of error that passes the test
 */
void ASSERT_ARRAY_RELATIVE_RANGE(float* expected, float* actual, int n, float range);

/**
 * \brief Asserts that actual=value for every array element
 * \param[in] actual	Array with actual values
 * \param[in] value		Expected value
 * \param[in] n			Array element count
 */
void ASSERT_ARRAY_EQ(float* actual, float value, int n);

/**
 * \brief Executes the call, synchronizes the device and puts the ellapsed time into 'time'.
 * \param[in] call	The call to be executed
 * \param[in] time	Measured time will be written here
 */
#define CUDA_MEASURE_TIME(call, time) \
		{ \
			cudaEvent_t start, stop; \
			cudaEventCreate(&start); \
			cudaEventCreate(&stop); \
			cudaEventRecord(start); \
			call; \
			cudaDeviceSynchronize(); \
			cudaEventRecord(stop); \
			cudaEventSynchronize(stop); \
			cudaEventElapsedTime(&time, start, stop); \
		}

/**
 * \brief Retrieves the file size in bytes.
 * \param[in] path	File path
 * \returns	File size in bytes
 */
int GetFileSize(string path);

/**
 * \brief Allocates an array in host memory that is a copy of the array in device memory.
 * \param[in] d_array	Array in device memory to be copied
 * \param[in] size		Array size in bytes
 * \returns Array pointer in host memory
 */
void* MallocFromDeviceArray(void* d_array, int size);

/**
 * \brief Allocates an array in device memory that is a copy of the array in host memory.
 * \param[in] h_array	Array in host memory to be copied
 * \param[in] size		Array size in bytes
 * \returns Array pointer in device memory
 */
void* CudaMallocFromHostArray(void* h_array, int size);

/**
 * \brief Creates an array in device memory that is filled with a binary file's content and has the same size.
 * \param[in] path	File path
 * \returns Array pointer in device memory
 */
void* MallocFromBinary(string path);

/**
 * \brief Creates an array in host memory that is filled with a binary file's content and has the same size.
 * \param[in] path	File path
 * \returns Array pointer in host memory
 */
void* CudaMallocFromBinaryFile(string path);

/**
 * \brief Creates an array of floats initialized to 0.0f in device memory with the specified element count.
 * \param[in] elements	Element count
 * \returns Array pointer in device memory
 */
float* CudaMallocZeroFilledFloat(int elements);

/**
 * \brief Calculates the mean difference between expected and actual values.
 * \param[in] expected	Array with expected values
 * \param[in] actual	Array with actual values
 * \param[in] n			Array element count
 * \returns Mean difference
 */
double GetMeanAbsoluteError(float* const expected, float* const actual, int n);

/**
 * \brief Calculates the mean quotient of difference between expected and actual values, and expected values, i. e. |(a-b)/a|. Cases with a=0 are left out.
 * \param[in] expected	Array with expected values
 * \param[in] actual	Array with actual values
 * \param[in] n			Array element count
 * \returns Mean quotient; or, in case every expected element is 0: -1
 */
double GetMeanRelativeError(float* const expected, float* const actual, int n);