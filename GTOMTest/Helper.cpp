#include "Prerequisites.h"

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

double GetMeanAbsoluteError(float* const expected, float* const actual, int n)
{
	double sum = 0.0;
	for(int i = 0; i < n; i++)
		sum += abs((double)expected[i] - (double)actual[i]);
	sum /= (double)n;

	return sum;
}

double GetMeanRelativeError(float* const expected, float* const actual, int n)
{
	double sum = 0.0;
	for(int i = 0; i < n; i++)
	{
		if(expected[i] == 0.0f)
			n--;
		else
			sum += abs((double)expected[i] - (double)actual[i]) / (double)expected[i];
	}

	if(n == 0)
		return -1.0;
	else
		return sum / (double)n;
}