#include "..\Prerequisites.cuh"
#include "..\Functions.cuh"

////////////////////////////
//CUDA kernel declarations//
////////////////////////////

template <class T> __global__ void RemapFullToHalfFFTKernel(T* d_input, T* d_output, int3 dims);
template <class T> __global__ void RemapHalfToHalfFFTKernel(T* d_input, T* d_output, int3 dims);
template <class T> __global__ void RemapHalfFFTToHalfKernel(T* d_input, T* d_output, int3 dims);


////////////////
//Host methods//
////////////////

template <class T> void d_RemapFullToHalfFFT(T* d_input, T* d_output, int3 dims)
{
	int TpB = min(256, NextMultipleOf(dims.x / 2 + 1, 32));
	dim3 grid = dim3(((dims.x / 2 + 1) + TpB - 1) / TpB, dims.y, dims.z);
	RemapFullToHalfFFTKernel <<<grid, TpB>>> (d_input, d_output, dims);
}
template void d_RemapFullToHalfFFT<tfloat>(tfloat* d_input, tfloat* d_output, int3 dims);
template void d_RemapFullToHalfFFT<tcomplex>(tcomplex* d_input, tcomplex* d_output, int3 dims);

template <class T> void d_RemapHalfToHalfFFT(T* d_input, T* d_output, int3 dims)
{
	int TpB = min(256, NextMultipleOf(dims.x / 2 + 1, 32));
	dim3 grid = dim3(((dims.x / 2 + 1) + TpB - 1) / TpB, dims.y, dims.z);
	RemapHalfToHalfFFTKernel <<<grid, TpB>>> (d_input, d_output, dims);
}
template void d_RemapHalfToHalfFFT<tfloat>(tfloat* d_input, tfloat* d_output, int3 dims);
template void d_RemapHalfToHalfFFT<tcomplex>(tcomplex* d_input, tcomplex* d_output, int3 dims);

template <class T> void d_RemapHalfFFTToHalf(T* d_input, T* d_output, int3 dims)
{
	int TpB = min(256, NextMultipleOf(dims.x / 2 + 1, 32));
	dim3 grid = dim3(((dims.x / 2 + 1) + TpB - 1) / TpB, dims.y, dims.z);
	RemapHalfFFTToHalfKernel <<<grid, TpB>>> (d_input, d_output, dims);
}
template void d_RemapHalfFFTToHalf<tfloat>(tfloat* d_input, tfloat* d_output, int3 dims);
template void d_RemapHalfFFTToHalf<tcomplex>(tcomplex* d_input, tcomplex* d_output, int3 dims);


////////////////
//CUDA kernels//
////////////////

template <class T> __global__ void RemapFullToHalfFFTKernel(T* d_input, T* d_output, int3 dims)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if(x >= dims.x / 2 + 1)
		return;

	int rx = (x + (dims.x / 2)) % dims.x;
	int ry = ((blockIdx.y + ((dims.y + 1) / 2)) % dims.y);
	int rz = ((blockIdx.z + ((dims.z + 1) / 2)) % dims.z);

	d_output[(rz * dims.y + ry) * (dims.x / 2 + 1) + x] = d_input[(blockIdx.z * dims.y + blockIdx.y) * dims.x + rx];
}

template <class T> __global__ void RemapHalfToHalfFFTKernel(T* d_input, T* d_output, int3 dims)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if(x >= dims.x / 2 + 1)
		return;

	int ry = ((blockIdx.y + ((dims.y + 1) / 2)) % dims.y);
	int rz = ((blockIdx.z + ((dims.z + 1) / 2)) % dims.z);

	d_output[(rz * dims.y + ry) * (dims.x / 2 + 1) + x] = d_input[(blockIdx.z * dims.y + blockIdx.y) * (dims.x / 2 + 1) + x];
}

template <class T> __global__ void RemapHalfFFTToHalfKernel(T* d_input, T* d_output, int3 dims)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if(x >= dims.x / 2 + 1)
		return;

	int ry = ((blockIdx.y + ((dims.y + 1) / 2)) % dims.y);
	int rz = ((blockIdx.z + ((dims.z + 1) / 2)) % dims.z);

	d_output[(blockIdx.z * dims.y + blockIdx.y) * (dims.x / 2 + 1) + x] = d_input[(rz * dims.y + ry) * (dims.x / 2 + 1) + x];
}