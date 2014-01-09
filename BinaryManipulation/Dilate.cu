#include "../Prerequisites.cuh"
#include "../Functions.cuh"

////////////////////////////
//CUDA kernel declarations//
////////////////////////////

template <class T> __global__ void Dilate2DKernel(T* d_input, T* d_output, int3 dims);


//////////
//Dilate//
//////////

template <class T> void d_Dilate(T* d_input, T* d_output, int3 dims, int batch)
{
	size_t TpB = min(192, NextMultipleOf(dims.x, 32));
	dim3 grid = dim3((dims.x + TpB - 1) / TpB, dims.y);
	Dilate2DKernel <<<grid, TpB>>> (d_input, d_output, dims);
}
template void d_Dilate<char>(char* d_input, char* d_output, int3 dims, int batch);
template void d_Dilate<int>(int* d_input, int* d_output, int3 dims, int batch);


////////////////
//CUDA kernels//
////////////////

template <class T> __global__ void Dilate2DKernel(T* d_input, T* d_output, int3 dims)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= dims.x)
		return;

	if(idx > 0 && d_input[blockIdx.y * dims.x + idx - 1])
	{
		d_output[blockIdx.y * dims.x + idx] = (T)1;
		return;
	}

	if(blockIdx.y > 0 && d_input[(blockIdx.y - 1) * dims.x + idx])
	{
		d_output[blockIdx.y * dims.x + idx] = (T)1;
		return;
	}

	if(idx < dims.x - 1 && d_input[blockIdx.y * dims.x + idx + 1])
	{
		d_output[blockIdx.y * dims.x + idx] = (T)1;
		return;
	}

	if(blockIdx.y < dims.y - 1 && d_input[(blockIdx.y + 1) * dims.x + idx])
	{
		d_output[blockIdx.y * dims.x + idx] = (T)1;
		return;
	}
}