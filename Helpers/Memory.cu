#include "..\Prerequisites.cuh"
#include "..\Functions.cuh"

void* MallocFromDeviceArray(void* d_array, size_t size)
{
	void* h_array = malloc(size);
	cudaMemcpy(h_array, d_array, size, cudaMemcpyDeviceToHost);

	return h_array;
}

void* CudaMallocFromHostArray(void* h_array, size_t size)
{
	void* d_array;
	cudaMalloc((void**)&d_array, size);
	cudaMemcpy(d_array, h_array, size, cudaMemcpyHostToDevice);

	return d_array;
}

void* CudaMallocFromHostArray(void* h_array, size_t devicesize, size_t hostsize)
{
	void* d_array;
	cudaMalloc((void**)&d_array, devicesize);
	cudaMemcpy(d_array, h_array, hostsize, cudaMemcpyHostToDevice);

	return d_array;
}

tfloat* MallocZeroFilledFloat(size_t elements)
{
	tfloat* h_array = (tfloat*)malloc(elements * sizeof(tfloat));

	intptr_t s_elements = (intptr_t)elements;
	#pragma omp for schedule(dynamic, 1024)
	for(intptr_t i = 0; i < s_elements; i++)
		h_array[i] = 0.0f;

	return h_array;
}

tfloat* CudaMallocZeroFilledFloat(size_t elements)
{
	tfloat* h_array = MallocZeroFilledFloat(elements);
	tfloat* d_array = (tfloat*)CudaMallocFromHostArray(h_array, elements * sizeof(tfloat));
	free(h_array);

	return d_array;
}