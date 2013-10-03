#include "Prerequisites.h"

void ASSERT_ARRAY_ABSOLUTE_RANGE(tfloat* expected, tfloat* actual, size_t n, tfloat range)
{
	int fails = 0;
	#pragma omp for schedule(dynamic, 1024)
	for(int i = 0; i < n; i++)
		if(abs(expected[i] - actual[i]) > range)
			#pragma omp atomic
			fails++;

	if(fails > 0)
		GTEST_CHECK_(false) << fails << " out of " << n << " elements had an absolute error of over " << range;
}

void ASSERT_ARRAY_RELATIVE_RANGE(tfloat* expected, tfloat* actual, size_t n, tfloat range)
{
	int fails = 0;
	#pragma omp for schedule(dynamic, 1024)
	for(int i = 0; i < n; i++)
		if(expected[i] == 0.0f && actual[i] == 0.0f)
			#pragma omp atomic
			fails++;
		else if(expected[i] != 0.0f && actual[i] != 0.0f && abs((actual[i] - expected[i]) / expected[i]) > range)
			#pragma omp atomic
			fails++;

	if(fails > 0)
		GTEST_CHECK_(false) << fails << " out of " << n << " elements had a relative error of over " << range;
}

template <class T> void ASSERT_ARRAY_EQ(T* actual, T value, size_t n)
{
	int fails = 0;
	#pragma omp for schedule(dynamic, 1024)
	for(intptr_t i = 0; i < n; i++)
		if(actual[i] != value)
			#pragma omp atomic
			fails++;

	if(fails > 0)
		GTEST_CHECK_(false) << fails << " out of " << n << " elements were not equal to " << value;
}
template void ASSERT_ARRAY_EQ<tfloat>(tfloat* actual, tfloat value, size_t n);
template void ASSERT_ARRAY_EQ<int>(int* actual, int value, size_t n);

template <class T> void ASSERT_ARRAY_EQ(T* actual, T* expected, size_t n)
{
	int fails = 0;
	#pragma omp for schedule(dynamic, 1024)
	for(intptr_t i = 0; i < n; i++)
		if(actual[i] != expected[i])
			#pragma omp atomic
			fails++;

	if(fails > 0)
		GTEST_CHECK_(false) << fails << " out of " << n << " elements were not equal to expected";
}
template void ASSERT_ARRAY_EQ<tfloat>(tfloat* actual, tfloat* expected, size_t n);
template void ASSERT_ARRAY_EQ<int>(int* actual, int* expected, size_t n);

int GetFileSize(string path)
{
	ifstream inputfile(path, ios::in|ios::binary|ios::ate);
	int size = inputfile.tellg();
	inputfile.close();

	return size;
}

void* CudaMallocFromBinaryFile(string path)
{
	void* h_array = MallocFromBinaryFile(path);
	void* d_array = CudaMallocFromHostArray(h_array, GetFileSize(path));
	free(h_array);

	return d_array;
}

void* MallocFromBinaryFile(string path)
{
	ifstream inputfile(path, ios::in|ios::binary|ios::ate);
	int size = inputfile.tellg();
	void* output = malloc(size);
	inputfile.seekg(0, ios::beg);
	inputfile.read((char*)output, size);
	inputfile.close();

	return output;
}

double GetMeanAbsoluteError(tfloat* const expected, tfloat* const actual, size_t n)
{
	double sum = 0.0;

	double* summands = (double*)malloc(n * sizeof(double));
	intptr_t s_n = (intptr_t)n;
	#pragma omp for schedule(dynamic, 1024)
	for(intptr_t i = 0; i < s_n; i++)
		summands[i] = abs((double)expected[i] - (double)actual[i]);

	double c = 0, y, t;
	for(size_t i = 0; i < n; i++)
	{
		y = summands[i] - c;
		t = sum + y;
		c = (t - sum) - y;
		sum = t;
	}

	free(summands);

	sum /= (double)n;

	return sum;
}

double GetMeanRelativeError(tfloat* const expected, tfloat* const actual, size_t n)
{
	double sum = 0.0;

	double* summands = (double*)malloc(n * sizeof(double));
	intptr_t s_n = (intptr_t)n;
	size_t actual_n = n;
	#pragma omp for schedule(dynamic, 1024)
	for(intptr_t i = 0; i < s_n; i++)
		if(expected[i] == 0.0f)
		{
			summands[i] = 0;
			#pragma omp atomic
			actual_n--;
		}
		else
			summands[i] = abs(((double)expected[i] - (double)actual[i]) / (double)expected[i]);

	double c = 0, y, t;
	for(size_t i = 0; i < n; i++)
	{
		y = summands[i] - c;
		t = sum + y;
		c = (t - sum) - y;
		sum = t;
	}

	if(actual_n == 0)
		return -1.0;
	else
		return sum / (double)actual_n;
}