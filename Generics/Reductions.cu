#include "../Prerequisites.cuh"
#include "../Functions.cuh"


////////////////////////////
//CUDA kernel declarations//
////////////////////////////

template<class T> __global__ void ReduceAddKernel(T* d_input, T* d_output, int nvectors);


//////////////////
//Multiplication//
//////////////////




//////////////////////////
//Complex Multiplication//
//////////////////////////




////////////
//Addition//
////////////

template<class T> void d_ReduceAdd(T* d_input, T* d_output, int vectorlength, int nvectors, int batch)
{
	int TpB = min(NextMultipleOf(nvectors, 32), 256);
	dim3 grid = dim3(vectorlength, batch);
	ReduceAddKernel<T> <<<grid, TpB>>> (d_input, d_output, nvectors);
}
template void d_ReduceAdd<char>(char* d_input, char* d_output, int vectorlength, int nvectors, int batch);
template void d_ReduceAdd<short>(short* d_input, short* d_output, int vectorlength, int nvectors, int batch);
template void d_ReduceAdd<int>(int* d_input, int* d_output, int vectorlength, int nvectors, int batch);
template void d_ReduceAdd<uint>(uint* d_input, uint* d_output, int vectorlength, int nvectors, int batch);
template void d_ReduceAdd<float>(float* d_input, float* d_output, int vectorlength, int nvectors, int batch);
template void d_ReduceAdd<double>(double* d_input, double* d_output, int vectorlength, int nvectors, int batch);

template<class T> __global__ void ReduceAddKernel(T* d_input, T* d_output, int nvectors)
{
	d_input += blockIdx.y * nvectors * gridDim.x + blockIdx.x;
	
	__shared__ T sums[256];

	T sum = (T)0;
	for (int n = threadIdx.x; n < nvectors; n += blockDim.x)
		sum += d_input[n * gridDim.x];

	sums[threadIdx.x] = sum;

	__syncthreads();

	if(threadIdx.x == 0)
	{
		for (int t = 1; t < min(nvectors, blockDim.x); t++)
			sum += sums[t];

		d_output[blockIdx.y * gridDim.x + blockIdx.x] = sum;
	}
}