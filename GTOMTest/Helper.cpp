#include "Prerequisites.h"

void ASSERT_ARRAY_ABSOLUTE_RANGE(float* expected, float* actual, int n, float range)
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

void ASSERT_ARRAY_RELATIVE_RANGE(float* expected, float* actual, int n, float range)
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

void ASSERT_ARRAY_EQ(float* actual, float value, int n)
{
	int fails = 0;
	#pragma omp for schedule(dynamic, 1024)
	for(int i = 0; i < n; i++)
		if(actual[i] != value)
			#pragma omp atomic
			fails++;

	if(fails > 0)
		GTEST_CHECK_(false) << fails << " out of " << n << " elements were not equal to " << value;
}

int GetFileSize(string path)
{
	ifstream inputfile(path, ios::in|ios::binary|ios::ate);
	int size = inputfile.tellg();
	inputfile.close();

	return size;
}

void* MallocFromDeviceArray(void* d_array, int size)
{
	void* h_array = malloc(size);
	cudaMemcpy(h_array, d_array, size, cudaMemcpyDeviceToHost);

	return h_array;
}

void* CudaMallocFromHostArray(void* h_array, int size)
{
	void* d_array;
	cudaMalloc((void**)&d_array, size);
	cudaMemcpy(d_array, h_array, size, cudaMemcpyHostToDevice);

	return d_array;
}

void* CudaMallocFromBinaryFile(string path)
{
	void* h_array = MallocFromBinary(path);
	void* d_array = CudaMallocFromHostArray(h_array, GetFileSize(path));
	free(h_array);

	return d_array;
}

void* MallocFromBinary(string path)
{
	ifstream inputfile(path, ios::in|ios::binary|ios::ate);
	int size = inputfile.tellg();
	void* output = malloc(size);
	inputfile.seekg(0, ios::beg);
	inputfile.read((char*)output, size);
	inputfile.close();

	return output;
}

float* CudaMallocZeroFilledFloat(int elements)
{
	float* h_array = (float*)malloc(elements * sizeof(float));

	#pragma omp for schedule(dynamic, 1024)
	for(int i = 0; i < elements; i++)
		h_array[i] = 0.0f;

	float* d_array = (float*)CudaMallocFromHostArray(h_array, elements * sizeof(float));
	free(h_array);

	return d_array;
}

double GetMeanAbsoluteError(float* const expected, float* const actual, int n)
{
	double sum = 0.0;

	#pragma omp for schedule(dynamic, 1024)
	for(int i = 0; i < n; i++)
	{
		float value = abs((double)expected[i] - (double)actual[i]);
		#pragma omp atomic
		sum += value;
	}
	sum /= (double)n;

	return sum;
}

double GetMeanRelativeError(float* const expected, float* const actual, int n)
{
	double sum = 0.0;

	#pragma omp for schedule(dynamic, 1024)
	for(int i = 0; i < n; i++)
	{
		if(expected[i] == 0.0f)
			#pragma omp atomic
			n--;
		else
		{
			float value = abs(((double)expected[i] - (double)actual[i]) / (double)expected[i]);
			#pragma omp atomic
			sum += value;
		}
	}

	if(n == 0)
		return -1.0;
	else
		return sum / (double)n;
}