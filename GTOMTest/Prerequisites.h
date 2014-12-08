#pragma once

#include "gtest/gtest.h"
#include "../include/GTOM.cuh"

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
void ASSERT_ARRAY_ABSOLUTE_RANGE(tfloat* expected, tfloat* actual, size_t n, tfloat range);

/**
 * \brief Asserts that |(expected-actual)/expected| is below the specified margin for every value pair
 * \param[in] expected	Array with expected values
 * \param[in] actual	Array with actual values
 * \param[in] n			Array element count
 * \param[in] range		Maximum margin of error that passes the test
 */
void ASSERT_ARRAY_RELATIVE_RANGE(tfloat* expected, tfloat* actual, size_t n, tfloat range);

/**
 * \brief Asserts that actual=value for every array element
 * \param[in] actual	Array with actual values
 * \param[in] value		Expected value
 * \param[in] n			Array element count
 */
template <class T> void ASSERT_ARRAY_EQ(T* actual, T value, size_t n);

/**
 * \brief Asserts that actual=value for every array element
 * \param[in] actual	Array with actual values
 * \param[in] value		Array with expected values
 * \param[in] n			Array element count
 */
template <class T> void ASSERT_ARRAY_EQ(T* actual, T* values, size_t n);

/**
 * \brief Retrieves the file size in bytes.
 * \param[in] path	File path
 * \returns	File size in bytes
 */
int GetFileSize(string path);

/**
 * \brief Creates an array in device memory that is filled with a binary file's content and has the same size.
 * \param[in] path	File path
 * \returns Array pointer in device memory
 */
void* MallocFromBinaryFile(string path);

void WriteToBinaryFile(string path, void* data, size_t bytes);

/**
 * \brief Creates an array in host memory that is filled with a binary file's content and has the same size.
 * \param[in] path	File path
 * \returns Array pointer in host memory
 */
void* CudaMallocFromBinaryFile(string path);

void CudaWriteToBinaryFile(string path, void* d_data, size_t elements);

/**
 * \brief Calculates the mean difference between expected and actual values.
 * \param[in] expected	Array with expected values
 * \param[in] actual	Array with actual values
 * \param[in] n			Array element count
 * \returns Mean difference
 */
double GetMeanAbsoluteError(float* const expected, float* const actual, size_t n);

/**
 * \brief Calculates the mean quotient of difference between expected and actual values, and expected values, i. e. |(a-b)/a|. Cases with a=0 are left out.
 * \param[in] expected	Array with expected values
 * \param[in] actual	Array with actual values
 * \param[in] n			Array element count
 * \returns Mean quotient; or, in case every expected element is 0: -1
 */
double GetMeanRelativeError(float* const expected, float* const actual, size_t n);

/**
 * \brief Calculates the sum of the array's elements using Kahan's algorithm to minimize the error for large arrays.
 * \param[in] expected	Array with values to be summed
 * \param[in] n			Array element count
 * \returns Sum
 */
template<class T> T KahanSum(T *data, int size);