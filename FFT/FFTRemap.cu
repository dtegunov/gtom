#include "..\Prerequisites.cuh"
#include "..\Functions.cuh"

////////////////////////////
//CUDA kernel declarations//
////////////////////////////

template <class T> __global__ void RemapFull2HalfFFTKernel(T* d_input, T* d_output, int3 dims);
template <class T> __global__ void RemapFullFFT2FullKernel(T* d_input, T* d_output, int3 dims);
template <class T> __global__ void RemapFull2FullFFTKernel(T* d_input, T* d_output, int3 dims);


////////////////
//Host methods//
////////////////

template <class T> void d_RemapFull2HalfFFT(T* d_input, T* d_output, int3 dims)
{
	int TpB = min(256, NextMultipleOf(dims.x / 2 + 1, 32));
	dim3 grid = dim3(((dims.x / 2 + 1) + TpB - 1) / TpB, dims.y, dims.z);
	RemapFull2HalfFFTKernel <<<grid, TpB>>> (d_input, d_output, dims);
	cudaStreamQuery(0);
}
template void d_RemapFull2HalfFFT<tfloat>(tfloat* d_input, tfloat* d_output, int3 dims);
template void d_RemapFull2HalfFFT<tcomplex>(tcomplex* d_input, tcomplex* d_output, int3 dims);
template void d_RemapFull2HalfFFT<int>(int* d_input, int* d_output, int3 dims);

template <class T> void d_RemapFullFFT2Full(T* d_input, T* d_output, int3 dims, int batch)
{
	size_t elements = dims.x * dims.y * dims.z;
	T* d_intermediate = NULL;
	if(d_input == d_output)
		cudaMalloc((void**)&d_intermediate, elements * sizeof(T));

	int TpB = min(256, NextMultipleOf(dims.x, 32));
	dim3 grid = dim3((dims.x + TpB - 1) / TpB, dims.y, dims.z);
	if(d_input != d_output)
	{
		for(int b = 0; b < batch; b++)
			RemapFullFFT2FullKernel <<<grid, TpB>>> (d_input + elements * b, d_output + elements * b, dims);
		cudaStreamQuery(0);
	}
	else		
	{
		for(int b = 0; b < batch; b++)
		{
			RemapFullFFT2FullKernel <<<grid, TpB>>> (d_input + elements * b, d_intermediate, dims);
			cudaMemcpy(d_output + elements * b, d_intermediate, elements * sizeof(T), cudaMemcpyDeviceToDevice);
		}
		cudaStreamQuery(0);
	}
}
template void d_RemapFullFFT2Full<tfloat>(tfloat* d_input, tfloat* d_output, int3 dims, int batch);
template void d_RemapFullFFT2Full<tcomplex>(tcomplex* d_input, tcomplex* d_output, int3 dims, int batch);
template void d_RemapFullFFT2Full<int>(int* d_input, int* d_output, int3 dims, int batch);

template <class T> void d_RemapFull2FullFFT(T* d_input, T* d_output, int3 dims, int batch)
{
	size_t elements = dims.x * dims.y * dims.z;
	T* d_intermediate = NULL;
	if(d_input == d_output)
		cudaMalloc((void**)&d_intermediate, elements * sizeof(T));

	int TpB = min(256, NextMultipleOf(dims.x, 32));
	dim3 grid = dim3((dims.x + TpB - 1) / TpB, dims.y, dims.z);
	if(d_input != d_output)
	{
		for(int b = 0; b < batch; b++)
			RemapFull2FullFFTKernel <<<grid, TpB>>> (d_input + elements * b, d_output + elements * b, dims);
		cudaStreamQuery(0);
	}
	else		
	{
		for(int b = 0; b < batch; b++)
		{
			RemapFull2FullFFTKernel <<<grid, TpB>>> (d_input + elements * b, d_intermediate, dims);
			cudaMemcpy(d_output + elements * b, d_intermediate, elements * sizeof(T), cudaMemcpyDeviceToDevice);
		}
		cudaStreamQuery(0);
	}
}
template void d_RemapFull2FullFFT<tfloat>(tfloat* d_input, tfloat* d_output, int3 dims, int batch);
template void d_RemapFull2FullFFT<tcomplex>(tcomplex* d_input, tcomplex* d_output, int3 dims, int batch);
template void d_RemapFull2FullFFT<int>(int* d_input, int* d_output, int3 dims, int batch);


////////////////
//CUDA kernels//
////////////////

template <class T> __global__ void RemapFull2HalfFFTKernel(T* d_input, T* d_output, int3 dims)
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	if(x >= dims.x / 2 + 1)
		return;

	uint rx = (x + (dims.x / 2)) % dims.x;
	uint ry = ((blockIdx.y + ((dims.y + 1) / 2)) % dims.y);
	uint rz = ((blockIdx.z + ((dims.z + 1) / 2)) % dims.z);

	d_output[(rz * dims.y + ry) * (dims.x / 2 + 1) + x] = d_input[(blockIdx.z * dims.y + blockIdx.y) * dims.x + rx];
}

template <class T> __global__ void RemapFullFFT2FullKernel(T* d_input, T* d_output, int3 dims)
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	if(x >= dims.x)
		return;
	
	uint rx = ((x + dims.x / 2) % dims.x);
	uint ry = ((blockIdx.y + dims.y / 2) % dims.y);
	uint rz = ((blockIdx.z + dims.z / 2) % dims.z);

	d_output[(rz * dims.y + ry) * dims.x + rx] = d_input[(blockIdx.z * dims.y + blockIdx.y) * dims.x + x];
}

template <class T> __global__ void RemapFull2FullFFTKernel(T* d_input, T* d_output, int3 dims)
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	if(x >= dims.x)
		return;
	
	uint rx = ((x + (dims.x + 1) / 2) % dims.x);
	uint ry = ((blockIdx.y + (dims.y + 1) / 2) % dims.y);
	uint rz = ((blockIdx.z + (dims.z + 1) / 2) % dims.z);

	d_output[(rz * dims.y + ry) * dims.x + rx] = d_input[(blockIdx.z * dims.y + blockIdx.y) * dims.x + x];
}