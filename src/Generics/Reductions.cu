#include "Prerequisites.cuh"


////////////////////////////
//CUDA kernel declarations//
////////////////////////////

template<class T> __global__ void ReduceAddKernel(T* d_input, T* d_output, int nvectors, int vectorlength);
template<class T> __global__ void ReduceMeanKernel(T* d_input, T* d_output, int nvectors, int vectorlength);



////////////
//Addition//
////////////

template<class T> void d_ReduceAdd(T* d_input, T* d_output, int vectorlength, int nvectors, int batch)
{
	int TpB = min(NextMultipleOf(nvectors, 32), 256);
	dim3 grid = dim3(min(vectorlength, 2048), batch);
	ReduceAddKernel<T> <<<grid, TpB>>> (d_input, d_output, nvectors, vectorlength);
}
template void d_ReduceAdd<char>(char* d_input, char* d_output, int vectorlength, int nvectors, int batch);
template void d_ReduceAdd<short>(short* d_input, short* d_output, int vectorlength, int nvectors, int batch);
template void d_ReduceAdd<int>(int* d_input, int* d_output, int vectorlength, int nvectors, int batch);
template void d_ReduceAdd<uint>(uint* d_input, uint* d_output, int vectorlength, int nvectors, int batch);
template void d_ReduceAdd<float>(float* d_input, float* d_output, int vectorlength, int nvectors, int batch);
template void d_ReduceAdd<double>(double* d_input, double* d_output, int vectorlength, int nvectors, int batch);

template<class T> __global__ void ReduceAddKernel(T* d_input, T* d_output, int nvectors, int vectorlength)
{
	d_input += blockIdx.y * nvectors * vectorlength;

	for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < vectorlength; id += gridDim.x * blockDim.x)
	{
		T sum = (T)0;

		for (int n = 0; n < nvectors; n++)
			sum += d_input[n * vectorlength + id];

		d_output[blockIdx.y * vectorlength + id] = sum;
	}
}


////////
//Mean//
////////

template<class T> void d_ReduceMean(T* d_input, T* d_output, int vectorlength, int nvectors, int batch)
{
	int TpB = min(NextMultipleOf(nvectors, 32), 256);
	dim3 grid = dim3(min(vectorlength, 2048), batch);
	ReduceMeanKernel<T> << <grid, TpB >> > (d_input, d_output, nvectors, vectorlength);
}
template void d_ReduceMean<char>(char* d_input, char* d_output, int vectorlength, int nvectors, int batch);
template void d_ReduceMean<short>(short* d_input, short* d_output, int vectorlength, int nvectors, int batch);
template void d_ReduceMean<int>(int* d_input, int* d_output, int vectorlength, int nvectors, int batch);
template void d_ReduceMean<uint>(uint* d_input, uint* d_output, int vectorlength, int nvectors, int batch);
template void d_ReduceMean<float>(float* d_input, float* d_output, int vectorlength, int nvectors, int batch);
template void d_ReduceMean<double>(double* d_input, double* d_output, int vectorlength, int nvectors, int batch);

template<class T> __global__ void ReduceMeanKernel(T* d_input, T* d_output, int nvectors, int vectorlength)
{
	d_input += blockIdx.y * nvectors * vectorlength;

	for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < vectorlength; id += gridDim.x * blockDim.x)
	{
		T sum = (T)0;

		for (int n = 0; n < nvectors; n++)
			sum += d_input[n * vectorlength + id];

		d_output[blockIdx.y * vectorlength + id] = sum / (T)nvectors;
	}
}