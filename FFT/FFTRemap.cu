#include "..\Prerequisites.cuh"
#include "..\Functions.cuh"

////////////////////////////
//CUDA kernel declarations//
////////////////////////////

<<<<<<< HEAD
template <class T> __global__ void RemapFullToHalfFFTKernel(T* d_input, T* d_output, int3 dims);
template <class T> __global__ void RemapHalfToHalfFFTKernel(T* d_input, T* d_output, int3 dims);
template <class T> __global__ void RemapHalfFFTToHalfKernel(T* d_input, T* d_output, int3 dims);
=======
template <class T> __global__ void RemapFull2HalfFFTKernel(T* d_input, T* d_output, int3 dims);
template <class T> __global__ void RemapFullFFT2FullKernel(T* d_input, T* d_output, int3 dims);
>>>>>>> 77ee24d2625debc91b0cc36e1f8bdad326e7221b


////////////////
//Host methods//
////////////////

<<<<<<< HEAD
template <class T> void d_RemapFullToHalfFFT(T* d_input, T* d_output, int3 dims)
{
	int TpB = min(256, NextMultipleOf(dims.x / 2 + 1, 32));
=======
template <class T> void d_RemapFull2HalfFFT(T* d_input, T* d_output, int3 dims)
{
	int TpB = min(256, dims.x / 2 + 1);
>>>>>>> 77ee24d2625debc91b0cc36e1f8bdad326e7221b
	dim3 grid = dim3(((dims.x / 2 + 1) + TpB - 1) / TpB, dims.y, dims.z);
	RemapFull2HalfFFTKernel <<<grid, TpB>>> (d_input, d_output, dims);
}
<<<<<<< HEAD
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
=======
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
>>>>>>> 77ee24d2625debc91b0cc36e1f8bdad326e7221b


////////////////
//CUDA kernels//
////////////////

<<<<<<< HEAD
template <class T> __global__ void RemapFullToHalfFFTKernel(T* d_input, T* d_output, int3 dims)
=======
template <class T> __global__ void RemapFull2HalfFFTKernel(T* d_input, T* d_output, int3 dims)
>>>>>>> 77ee24d2625debc91b0cc36e1f8bdad326e7221b
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if(x >= dims.x / 2 + 1)
		return;

	int rx = (x + (dims.x / 2)) % dims.x;
	int ry = ((blockIdx.y + ((dims.y + 1) / 2)) % dims.y);
	int rz = ((blockIdx.z + ((dims.z + 1) / 2)) % dims.z);

	d_output[(rz * dims.y + ry) * (dims.x / 2 + 1) + x] = d_input[(blockIdx.z * dims.y + blockIdx.y) * dims.x + rx];
}

<<<<<<< HEAD
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
=======
template <class T> __global__ void RemapFullFFT2FullKernel(T* d_input, T* d_output, int3 dims)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if(x >= dims.x)
		return;
	
	int rx = ((x + dims.x / 2) % dims.x);
	int ry = ((blockIdx.y + dims.y / 2) % dims.y);
	int rz = ((blockIdx.z + dims.z / 2) % dims.z);

	d_output[(rz * dims.y + ry) * dims.x + rx] = d_input[(blockIdx.z * dims.y + blockIdx.y) * dims.x + x];
>>>>>>> 77ee24d2625debc91b0cc36e1f8bdad326e7221b
}