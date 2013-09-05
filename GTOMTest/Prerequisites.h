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

#define ASSERT_ABSOLUTE_RANGE(expected, actual, range) \
		GTEST_CHECK_(abs(actual - expected) <= range) \
		<< "\nExpected: " << expected << ", Actual: " << actual \
		<< ", Difference: " << abs(expected - actual)

#define ASSERT_RELATIVE_RANGE(expected, actual, range) \
		if (expected == 0.0f && actual != 0.0f) \
		{ \
			GTEST_CHECK_(false) << "\nExpected: " << expected << ", Actual: " << actual; \
		} \
		else if (expected != 0.0f && actual != 0.0f) \
		{ \
			GTEST_CHECK_(abs(actual - expected) / expected <= range) \
			<< "\nExpected: " << expected << ", Actual: " << actual \
			<< ", Relative difference: " << (abs(expected - actual) / expected); \
		}\

char* LoadArrayFromBinary(string path);

double GetMeanAbsoluteError(float* const expected, float* const actual, int n);
double GetMeanRelativeError(float* const expected, float* const actual, int n);