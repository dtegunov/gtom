#include "Prerequisites.cuh"
#include "FFT.cuh"

namespace gtom
{
	////////////////////////////
	//CUDA kernel declarations//
	////////////////////////////

	template <class T> __global__ void RemapFull2HalfFFTKernel(T* d_input, T* d_output, int3 dims);
	template <class T> __global__ void RemapFullFFT2FullKernel(T* d_input, T* d_output, uint3 dims, uint elements);
	template <class T> __global__ void RemapFull2FullFFTKernel(T* d_input, T* d_output, uint3 dims, uint elements);
	template <class T> __global__ void RemapHalfFFT2HalfKernel(T* d_input, T* d_output, int3 dims);
	template <class T> __global__ void RemapHalf2HalfFFTKernel(T* d_input, T* d_output, int3 dims);


	////////////////
	//Host methods//
	////////////////

	template <class T> void d_RemapFull2HalfFFT(T* d_input, T* d_output, int3 dims, int batch)
	{
		T* d_intermediate = NULL;
		if (d_input == d_output)
			cudaMalloc((void**)&d_intermediate, ElementsFFT(dims) * batch * sizeof(T));
		else
			d_intermediate = d_output;

		int TpB = min(256, NextMultipleOf(dims.x / 2 + 1, 32));
		dim3 grid = dim3(dims.y, dims.z, batch);
		RemapFull2HalfFFTKernel << <grid, TpB >> > (d_input, d_intermediate, dims);

		if (d_input == d_output)
		{
			cudaMemcpy(d_output, d_intermediate, ElementsFFT(dims) * batch * sizeof(T), cudaMemcpyDeviceToDevice);
			cudaFree(d_intermediate);
		}
	}
	template void d_RemapFull2HalfFFT<tfloat>(tfloat* d_input, tfloat* d_output, int3 dims, int batch);
	template void d_RemapFull2HalfFFT<tcomplex>(tcomplex* d_input, tcomplex* d_output, int3 dims, int batch);
	template void d_RemapFull2HalfFFT<int>(int* d_input, int* d_output, int3 dims, int batch);

	template <class T> void d_RemapFullFFT2Full(T* d_input, T* d_output, int3 dims, int batch)
	{
		T* d_intermediate = NULL;
		if (d_input == d_output)
			cudaMalloc((void**)&d_intermediate, Elements(dims) * batch * sizeof(T));
		else
			d_intermediate = d_output;

		int TpB = min(256, NextMultipleOf(dims.x, 32));
		dim3 grid = dim3(dims.y, dims.z, batch);
		RemapFullFFT2FullKernel << <grid, TpB >> > (d_input, d_intermediate, make_uint3(dims.x, dims.y, dims.z), Elements(dims));

		if (d_input == d_output)
		{
			cudaMemcpy(d_output, d_intermediate, Elements(dims) * batch * sizeof(T), cudaMemcpyDeviceToDevice);
			cudaFree(d_intermediate);
		}
	}
	template void d_RemapFullFFT2Full<tfloat>(tfloat* d_input, tfloat* d_output, int3 dims, int batch);
	template void d_RemapFullFFT2Full<tcomplex>(tcomplex* d_input, tcomplex* d_output, int3 dims, int batch);
	template void d_RemapFullFFT2Full<int>(int* d_input, int* d_output, int3 dims, int batch);

	template <class T> void d_RemapFull2FullFFT(T* d_input, T* d_output, int3 dims, int batch)
	{
		T* d_intermediate = NULL;
		if (d_input == d_output)
			cudaMalloc((void**)&d_intermediate, Elements(dims) * batch * sizeof(T));
		else
			d_intermediate = d_output;

		int TpB = min(256, NextMultipleOf(dims.x, 32));
		dim3 grid = dim3(dims.y, dims.z, batch);
		RemapFull2FullFFTKernel << <grid, TpB >> > (d_input, d_intermediate, make_uint3(dims.x, dims.y, dims.z), Elements(dims));

		if (d_input == d_output)
		{
			cudaMemcpy(d_output, d_intermediate, Elements(dims) * batch * sizeof(T), cudaMemcpyDeviceToDevice);
			cudaFree(d_intermediate);
		}
	}
	template void d_RemapFull2FullFFT<tfloat>(tfloat* d_input, tfloat* d_output, int3 dims, int batch);
	template void d_RemapFull2FullFFT<tcomplex>(tcomplex* d_input, tcomplex* d_output, int3 dims, int batch);
	template void d_RemapFull2FullFFT<int>(int* d_input, int* d_output, int3 dims, int batch);

	template <class T> void d_RemapHalfFFT2Half(T* d_input, T* d_output, int3 dims, int batch)
	{
		T* d_intermediate = NULL;
		if (d_input == d_output)
			cudaMalloc((void**)&d_intermediate, ElementsFFT(dims) * batch * sizeof(T));
		else
			d_intermediate = d_output;

		int TpB = min(256, NextMultipleOf(dims.x / 2 + 1, 32));
		dim3 grid = dim3(dims.y, dims.z, batch);
		RemapHalfFFT2HalfKernel << <grid, TpB >> > (d_input, d_intermediate, dims);

		if (d_input == d_output)
		{
			cudaMemcpy(d_output, d_intermediate, ElementsFFT(dims) * batch * sizeof(T), cudaMemcpyDeviceToDevice);
			cudaFree(d_intermediate);
		}
	}
	template void d_RemapHalfFFT2Half<tfloat>(tfloat* d_input, tfloat* d_output, int3 dims, int batch);
	template void d_RemapHalfFFT2Half<tcomplex>(tcomplex* d_input, tcomplex* d_output, int3 dims, int batch);
	template void d_RemapHalfFFT2Half<int>(int* d_input, int* d_output, int3 dims, int batch);

	template <class T> void d_RemapHalf2HalfFFT(T* d_input, T* d_output, int3 dims, int batch)
	{
		T* d_intermediate = NULL;
		if (d_input == d_output)
			cudaMalloc((void**)&d_intermediate, ElementsFFT(dims) * batch * sizeof(T));
		else
			d_intermediate = d_output;

		int TpB = min(256, NextMultipleOf(dims.x / 2 + 1, 32));
		dim3 grid = dim3(dims.y, dims.z, batch);
		RemapHalf2HalfFFTKernel << <grid, TpB >> > (d_input, d_intermediate, dims);

		if (d_input == d_output)
		{
			cudaMemcpy(d_output, d_intermediate, ElementsFFT(dims) * batch * sizeof(T), cudaMemcpyDeviceToDevice);
			cudaFree(d_intermediate);
		}
	}
	template void d_RemapHalf2HalfFFT<tfloat>(tfloat* d_input, tfloat* d_output, int3 dims, int batch);
	template void d_RemapHalf2HalfFFT<tcomplex>(tcomplex* d_input, tcomplex* d_output, int3 dims, int batch);
	template void d_RemapHalf2HalfFFT<int>(int* d_input, int* d_output, int3 dims, int batch);


	////////////////
	//CUDA kernels//
	////////////////

	template <class T> __global__ void RemapFull2HalfFFTKernel(T* d_input, T* d_output, int3 dims)
	{
		d_input += Elements(dims) * blockIdx.z;
		d_output += Elements(dims) * blockIdx.z;

		for (uint x = threadIdx.x; x < dims.x / 2 + 1; x += blockDim.x)
		{
			uint rx = (x + (dims.x / 2)) % dims.x;
			uint ry = ((blockIdx.x + ((dims.y + 1) / 2)) % dims.y);
			uint rz = ((blockIdx.y + ((dims.z + 1) / 2)) % dims.z);

			d_output[(rz * dims.y + ry) * (dims.x / 2 + 1) + x] = d_input[(blockIdx.y * dims.y + blockIdx.x) * dims.x + rx];
		}
	}

	template <class T> __global__ void RemapFullFFT2FullKernel(T* d_input, T* d_output, uint3 dims, uint elements)
	{
		uint ry = FFTShift(blockIdx.x, dims.x);
		uint rz = FFTShift(blockIdx.y, dims.z);

		d_output += elements * blockIdx.z + (rz * dims.y + ry) * dims.x;
		d_input += elements * blockIdx.z + (blockIdx.y * dims.y + blockIdx.x) * dims.x;

		for (uint x = threadIdx.x; x < dims.x; x += blockDim.x)
		{
			uint rx = FFTShift(x, dims.x);
			d_output[rx] = d_input[x];
		}
	}

	template <class T> __global__ void RemapFull2FullFFTKernel(T* d_input, T* d_output, uint3 dims, uint elements)
	{
		uint ry = IFFTShift(blockIdx.x, dims.y);
		uint rz = IFFTShift(blockIdx.y, dims.z);

		d_output += elements * blockIdx.z + (rz * dims.y + ry) * dims.x;
		d_input += elements * blockIdx.z + (blockIdx.y * dims.y + blockIdx.x) * dims.x;

		for (uint x = threadIdx.x; x < dims.x; x += blockDim.x)
		{
			uint rx = IFFTShift(x, dims.x);
			d_output[rx] = d_input[x];
		}
	}

	template <class T> __global__ void RemapHalfFFT2HalfKernel(T* d_input, T* d_output, int3 dims)
	{
		d_input += ElementsFFT(dims) * blockIdx.z;
		d_output += ElementsFFT(dims) * blockIdx.z;

		for (uint x = threadIdx.x; x < dims.x / 2 + 1; x += blockDim.x)
		{
			uint rx = dims.x / 2 - x;
			uint ry = dims.y - 1 - ((blockIdx.x + dims.y / 2 - 1) % dims.y);
			uint rz = dims.z - 1 - ((blockIdx.y + dims.z / 2 - 1) % dims.z);

			d_output[(rz * dims.y + ry) * (dims.x / 2 + 1) + rx] = d_input[(blockIdx.y * dims.y + blockIdx.x) * (dims.x / 2 + 1) + x];
		}
	}

	template <class T> __global__ void RemapHalf2HalfFFTKernel(T* d_input, T* d_output, int3 dims)
	{
		d_input += ElementsFFT(dims) * blockIdx.z;
		d_output += ElementsFFT(dims) * blockIdx.z;

		for (uint x = threadIdx.x; x < dims.x / 2 + 1; x += blockDim.x)
		{
			uint rx = dims.x / 2 - x;
			uint ry = dims.y - 1 - ((blockIdx.x + (dims.y + 1) / 2 - 1) % dims.y);
			uint rz = dims.z - 1 - ((blockIdx.y + (dims.z + 1) / 2 - 1) % dims.z);

			d_output[(rz * dims.y + ry) * (dims.x / 2 + 1) + rx] = d_input[(blockIdx.y * dims.y + blockIdx.x) * (dims.x / 2 + 1) + x];
		}
	}
}