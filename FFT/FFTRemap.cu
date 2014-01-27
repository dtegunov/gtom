#include "../Prerequisites.cuh"
#include "../Functions.cuh"

////////////////////////////
//CUDA kernel declarations//
////////////////////////////

template <class T> __global__ void RemapFull2HalfFFTKernel(T* d_input, T* d_output, int3 dims);
template <class T> __global__ void RemapFullFFT2FullKernel(T* d_input, T* d_output, int3 dims);
template <class T> __global__ void RemapFull2FullFFTKernel(T* d_input, T* d_output, int3 dims);


////////////////
//Host methods//
////////////////

template <class T> void d_RemapFull2HalfFFT(T* d_input, T* d_output, int3 dims, int batch)
{
	int TpB = min(256, NextMultipleOf(dims.x / 2 + 1, 32));
	dim3 grid = dim3(dims.y, dims.z, batch);
	RemapFull2HalfFFTKernel <<<grid, TpB>>> (d_input, d_output, dims);
	cudaStreamQuery(0);
}
template void d_RemapFull2HalfFFT<tfloat>(tfloat* d_input, tfloat* d_output, int3 dims, int batch);
template void d_RemapFull2HalfFFT<tcomplex>(tcomplex* d_input, tcomplex* d_output, int3 dims, int batch);
template void d_RemapFull2HalfFFT<int>(int* d_input, int* d_output, int3 dims, int batch);

template <class T> void d_RemapFullFFT2Full(T* d_input, T* d_output, int3 dims, int batch)
{
	size_t elements = dims.x * dims.y * dims.z;
	T* d_intermediate = NULL;
	if(d_input == d_output)
		cudaMalloc((void**)&d_intermediate, elements * batch * sizeof(T));

	int TpB = min(256, NextMultipleOf(dims.x, 32));
	dim3 grid = dim3(dims.y, dims.z, batch);
	if(d_input != d_output)
	{
		RemapFullFFT2FullKernel <<<grid, TpB>>> (d_input, d_output, dims);
		cudaStreamQuery(0);
	}
	else		
	{
		RemapFullFFT2FullKernel <<<grid, TpB>>> (d_input, d_intermediate, dims);
		cudaMemcpy(d_output, d_intermediate, elements * batch * sizeof(T), cudaMemcpyDeviceToDevice);
		cudaStreamQuery(0);
	}

	if(d_input == d_output)
		cudaFree(d_intermediate);
}
template void d_RemapFullFFT2Full<tfloat>(tfloat* d_input, tfloat* d_output, int3 dims, int batch);
template void d_RemapFullFFT2Full<tcomplex>(tcomplex* d_input, tcomplex* d_output, int3 dims, int batch);
template void d_RemapFullFFT2Full<int>(int* d_input, int* d_output, int3 dims, int batch);

template <class T> void d_RemapFull2FullFFT(T* d_input, T* d_output, int3 dims, int batch)
{
	size_t elements = dims.x * dims.y * dims.z;
	T* d_intermediate = NULL;
	if(d_input == d_output)
		cudaMalloc((void**)&d_intermediate, elements * batch * sizeof(T));

	int TpB = min(256, NextMultipleOf(dims.x, 32));
	dim3 grid = dim3(dims.y, dims.z, batch);
	if(d_input != d_output)
	{
		RemapFull2FullFFTKernel <<<grid, TpB>>> (d_input, d_output, dims);
		cudaStreamQuery(0);
	}
	else		
	{
		RemapFull2FullFFTKernel <<<grid, TpB>>> (d_input, d_intermediate, dims);
		cudaMemcpy(d_output, d_intermediate, elements * batch * sizeof(T), cudaMemcpyDeviceToDevice);
		cudaStreamQuery(0);
	}

	if(d_input == d_output)
		cudaFree(d_intermediate);
}
template void d_RemapFull2FullFFT<tfloat>(tfloat* d_input, tfloat* d_output, int3 dims, int batch);
template void d_RemapFull2FullFFT<tcomplex>(tcomplex* d_input, tcomplex* d_output, int3 dims, int batch);
template void d_RemapFull2FullFFT<int>(int* d_input, int* d_output, int3 dims, int batch);


////////////////
//CUDA kernels//
////////////////

template <class T> __global__ void RemapFull2HalfFFTKernel(T* d_input, T* d_output, int3 dims)
{
	d_input += Elements(dims) * blockIdx.z;
	d_output += Elements(dims) * blockIdx.z;

	for(uint x = threadIdx.x; x < dims.x / 2 + 1; x += blockDim.x)
	{
		uint rx = (x + (dims.x / 2)) % dims.x;
		uint ry = ((blockIdx.x + ((dims.y + 1) / 2)) % dims.y);
		uint rz = ((blockIdx.y + ((dims.z + 1) / 2)) % dims.z);

		d_output[(rz * dims.y + ry) * (dims.x / 2 + 1) + x] = d_input[(blockIdx.y * dims.y + blockIdx.x) * dims.x + rx];
	}
}

template <class T> __global__ void RemapFullFFT2FullKernel(T* d_input, T* d_output, int3 dims)
{
	d_input += Elements(dims) * blockIdx.z;
	d_output += Elements(dims) * blockIdx.z;

	for(uint x = threadIdx.x; x < dims.x; x += blockDim.x)
	{
		uint rx = ((x + dims.x / 2) % dims.x);
		uint ry = ((blockIdx.x + dims.y / 2) % dims.y);
		uint rz = ((blockIdx.y + dims.z / 2) % dims.z);

		d_output[(rz * dims.y + ry) * dims.x + rx] = d_input[(blockIdx.y * dims.y + blockIdx.x) * dims.x + x];
	}
}

template <class T> __global__ void RemapFull2FullFFTKernel(T* d_input, T* d_output, int3 dims)
{
	d_input += Elements(dims) * blockIdx.z;
	d_output += Elements(dims) * blockIdx.z;

	for(uint x = threadIdx.x; x < dims.x; x += blockDim.x)
	{
		uint rx = ((x + (dims.x + 1) / 2) % dims.x);
		uint ry = ((blockIdx.x + (dims.y + 1) / 2) % dims.y);
		uint rz = ((blockIdx.y + (dims.z + 1) / 2) % dims.z);

		d_output[(rz * dims.y + ry) * dims.x + rx] = d_input[(blockIdx.y * dims.y + blockIdx.x) * dims.x + x];
	}
}