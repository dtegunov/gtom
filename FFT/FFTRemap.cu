#include "..\Prerequisites.cuh"
#include "..\Functions.cuh"

////////////////////////////
//CUDA kernel declarations//
////////////////////////////

template <class T> __global__ void RemapFull2HalfFFTKernel(T* d_input, T* d_output, int3 dims);
template <class T> __global__ void RemapFullFFT2FullKernel(T* d_input, T* d_output, int3 dims);


////////////////
//Host methods//
////////////////

template <class T> void d_RemapFull2HalfFFT(T* d_input, T* d_output, int3 dims)
{
	int TpB = min(256, dims.x / 2 + 1);
	dim3 grid = dim3(((dims.x / 2 + 1) + TpB - 1) / TpB, dims.y, dims.z);
	RemapFull2HalfFFTKernel <<<grid, TpB>>> (d_input, d_output, dims);
}
template void d_RemapFull2HalfFFT<tfloat>(tfloat* d_input, tfloat* d_output, int3 dims);
template void d_RemapFull2HalfFFT<tcomplex>(tcomplex* d_input, tcomplex* d_output, int3 dims);
template void d_RemapFull2HalfFFT<int>(int* d_input, int* d_output, int3 dims);

template <class T> void d_RemapFullFFT2Full(T* d_input, T* d_output, int3 dims)
{
	int TpB = min(256, dims.x);
	dim3 grid = dim3((dims.x + TpB - 1) / TpB, dims.y, dims.z);
	RemapFullFFT2FullKernel <<<grid, TpB>>> (d_input, d_output, dims);
}
template void d_RemapFullFFT2Full<tfloat>(tfloat* d_input, tfloat* d_output, int3 dims);
template void d_RemapFullFFT2Full<tcomplex>(tcomplex* d_input, tcomplex* d_output, int3 dims);
template void d_RemapFullFFT2Full<int>(int* d_input, int* d_output, int3 dims);


////////////////
//CUDA kernels//
////////////////

template <class T> __global__ void RemapFull2HalfFFTKernel(T* d_input, T* d_output, int3 dims)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if(x >= dims.x / 2 + 1)
		return;

	int rx = (x + (dims.x / 2)) % dims.x;
	int ry = ((blockIdx.y + ((dims.y + 1) / 2)) % dims.y);
	int rz = ((blockIdx.z + ((dims.z + 1) / 2)) % dims.z);

	d_output[(rz * dims.y + ry) * (dims.x / 2 + 1) + x] = d_input[(blockIdx.z * dims.y + blockIdx.y) * dims.x + rx];
}

template <class T> __global__ void RemapFullFFT2FullKernel(T* d_input, T* d_output, int3 dims)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if(x >= dims.x)
		return;
	
	int rx = ((x + dims.x / 2) % dims.x);
	int ry = ((blockIdx.y + dims.y / 2) % dims.y);
	int rz = ((blockIdx.z + dims.z / 2) % dims.z);

	d_output[(rz * dims.y + ry) * dims.x + rx] = d_input[(blockIdx.z * dims.y + blockIdx.y) * dims.x + x];
}