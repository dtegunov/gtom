#include "Prerequisites.cuh"

#ifndef BlockSize
	#define BlockSize 1024
#endif


////////////////////////////
//CUDA kernel declarations//
////////////////////////////

template <class Tfrom, class Tto> __global__ void ConvertToKernel(Tfrom const* const d_original, Tto* const d_copy, size_t const n);
template <class T> __global__ void ConvertSplitComplexToTComplexKernel(T const* const d_originalr, T const* const d_originali, tcomplex* const d_copy, size_t const n);
template <class T> __global__ void ConvertTComplexToSplitComplexKernel(tcomplex const* const d_original, T* const d_copyr, T* const d_copyi, size_t const n);
template <class T> __global__ void ReKernel(tcomplex const* const d_input, T* const d_output, size_t const n);
template <class T> __global__ void ImKernel(tcomplex const* const d_input, T* const d_output, size_t const n);


////////////////////
//Host conversions//
////////////////////

template <class T> tfloat* ConvertToTFloat(T const* const original, size_t const n)
{
	tfloat* converted = (tfloat*)malloc(n * sizeof(tfloat));
	ConvertToTFloat<T>(original, converted, n);

	return converted;
}
template tfloat* ConvertToTFloat<double>(double const* const original, size_t const n);
template tfloat* ConvertToTFloat<float>(float const* const original, size_t const n);
template tfloat* ConvertToTFloat<int>(int const* const original, size_t const n);
template tfloat* ConvertToTFloat<uint>(uint const* const original, size_t const n);
template tfloat* ConvertToTFloat<short>(short const* const original, size_t const n);
template tfloat* ConvertToTFloat<ushort>(ushort const* const original, size_t const n);

template <class T> void ConvertToTFloat(T const* const original, tfloat* const copy, size_t const n)
{
	#pragma omp for schedule(dynamic, BlockSize)
	for(int i = 0; i < n; i++)
		copy[i] = (tfloat)original[i];
}
template void ConvertToTFloat<double>(double const* const original, tfloat* const copy, size_t const n);
template void ConvertToTFloat<float>(float const* const original, tfloat* const copy, size_t const n);
template void ConvertToTFloat<int>(int const* const original, tfloat* const copy, size_t const n);
template void ConvertToTFloat<uint>(uint const* const original, tfloat* const copy, size_t const n);
template void ConvertToTFloat<short>(short const* const original, tfloat* const copy, size_t const n);
template void ConvertToTFloat<ushort>(ushort const* const original, tfloat* const copy, size_t const n);

template <class T> T* ConvertTFloatTo(tfloat const* const original, size_t const n)
{
	T* converted = (T*)malloc(n * sizeof(T));
	ConvertTFloatTo<T>(original, converted, n);

	return converted;
}
template double* ConvertTFloatTo<double>(tfloat const* const original, size_t const n);
template float* ConvertTFloatTo<float>(tfloat const* const original, size_t const n);

template <class T> void ConvertTFloatTo(tfloat const* const original, T* const copy, size_t const n)
{
	#pragma omp for schedule(dynamic, BlockSize)
	for(int i = 0; i < n; i++)
		copy[i] = (T)original[i];
}
template void ConvertTFloatTo<double>(tfloat const* const original, double* const copy, size_t const n);
template void ConvertTFloatTo<float>(tfloat const* const original, float* const copy, size_t const n);

template <class T> tcomplex* ConvertSplitComplexToTComplex(T const* const originalr, T const* const originali, size_t const n)
{
	tcomplex* converted = (tcomplex*)malloc(n * sizeof(tcomplex));
	ConvertSplitComplexToTComplex(originalr, originali, converted, n);

	return converted;
}
template tcomplex* ConvertSplitComplexToTComplex<double>(double const* const originalr, double const* const originali, size_t const n);
template tcomplex* ConvertSplitComplexToTComplex<float>(float const* const originalr, float const* const originali, size_t const n);

template <class T> void ConvertSplitComplexToTComplex(T const* const originalr, T const* const originali, tcomplex* const copy, size_t const n)
{
	#pragma omp for schedule(dynamic, BlockSize)
	for(int i = 0; i < n; i++)
	{
		copy[i].x = (tfloat)originalr[i];
		copy[i].y = (tfloat)originali[i];
	}
}
template void ConvertSplitComplexToTComplex<double>(double const* const originalr, double const* const originali, tcomplex* const copy, size_t const n);
template void ConvertSplitComplexToTComplex<float>(float const* const originalr, float const* const originali, tcomplex* const copy, size_t const n);

template <class T> T* ConvertTComplexToSplitComplex(tcomplex const* const original, size_t const n)
{
	T* converted = (T*)malloc(n * 2 * sizeof(T));
	ConvertTComplexToSplitComplex<T>(original, converted, converted + n, n);

	return converted;
}
template double* ConvertTComplexToSplitComplex<double>(tcomplex const* const original, size_t const n);
template float* ConvertTComplexToSplitComplex<float>(tcomplex const* const original, size_t const n);

template <class T> void ConvertTComplexToSplitComplex(tcomplex const* const original, T* const copyr, T* const copyi, size_t const n)
{
	#pragma omp for schedule(dynamic, BlockSize)
	for(int i = 0; i < n; i++)
	{
		copyr[i] = (T)original[i].x;
		copyi[i] = (T)original[i].y;
	}
}
template void ConvertTComplexToSplitComplex<double>(tcomplex const* const original, double* const copyr, double* const copyi, size_t const n);
template void ConvertTComplexToSplitComplex<float>(tcomplex const* const original, float* const copyr, float* const copyi, size_t const n);


//////////////////////
//Device conversions//
//////////////////////

template <class T> void d_ConvertToTFloat(T const* const d_original, tfloat* const d_copy, size_t const n)
{
	size_t TpB = min((size_t)256, NextMultipleOf(n, 32));
	size_t totalblocks = min((n + TpB - 1) / TpB, (size_t)128);
	dim3 grid = dim3((uint)totalblocks);
	ConvertToKernel<T, tfloat> <<<grid, (uint)TpB>>> (d_original, d_copy, n);
}
template void d_ConvertToTFloat<double>(double const* const d_original, tfloat* const d_copy, size_t const n);
template void d_ConvertToTFloat<float>(float const* const d_original, tfloat* const d_copy, size_t const n);
template void d_ConvertToTFloat<int>(int const* const d_original, tfloat* const d_copy, size_t const n);
template void d_ConvertToTFloat<uint>(uint const* const d_original, tfloat* const d_copy, size_t const n);
template void d_ConvertToTFloat<short>(short const* const d_original, tfloat* const d_copy, size_t const n);
template void d_ConvertToTFloat<ushort>(ushort const* const d_original, tfloat* const d_copy, size_t const n);

template <class T> void d_ConvertTFloatTo(tfloat const* const d_original, T* const d_copy, size_t const n)
{
	size_t TpB = min((size_t)256, NextMultipleOf(n, 32));
	size_t totalblocks = min((n + TpB - 1) / TpB, (size_t)128);
	dim3 grid = dim3((uint)totalblocks);
	ConvertToKernel<tfloat, T> <<<grid, (uint)TpB>>> (d_original, d_copy, n);
}
template void d_ConvertTFloatTo<double>(tfloat const* const d_original, double* const d_copy, size_t const n);
template void d_ConvertTFloatTo<float>(tfloat const* const d_original, float* const d_copy, size_t const n);

template <class T> void d_ConvertSplitComplexToTComplex(T const* const d_originalr, T const* const d_originali, tcomplex* const d_copy, size_t const n)
{
	size_t TpB = min((size_t)256, NextMultipleOf(n, 32));
	size_t totalblocks = min((n + TpB - 1) / TpB, (size_t)128);
	dim3 grid = dim3((uint)totalblocks);
	ConvertSplitComplexToTComplexKernel<T> <<<grid, (uint)TpB>>> (d_originalr, d_originali, d_copy, n);
}
template void d_ConvertSplitComplexToTComplex<double>(double const* const d_originalr, double const* const d_originali, tcomplex* const d_copy, size_t const n);
template void d_ConvertSplitComplexToTComplex<float>(float const* const d_originalr, float const* const d_originali, tcomplex* const d_copy, size_t const n);

template <class T> void d_ConvertTComplexToSplitComplex(tcomplex const* const d_original, T* const d_copyr, T* const d_copyi, size_t const n)
{
	size_t TpB = min((size_t)256, NextMultipleOf(n, 32));
	size_t totalblocks = min((n + TpB - 1) / TpB, (size_t)128);
	dim3 grid = dim3((uint)totalblocks);
	ConvertTComplexToSplitComplexKernel<T> <<<grid, (uint)TpB>>> (d_original, d_copyr, d_copyi, n);
}
template void d_ConvertTComplexToSplitComplex<double>(tcomplex const* const d_original, double* const d_copyr, double* const d_copyi, size_t const n);
template void d_ConvertTComplexToSplitComplex<float>(tcomplex const* const d_original, float* const d_copyr, float* const d_copyi, size_t const n);

template <class T> void d_Re(tcomplex const* const d_input, T* const d_output, size_t const n)
{
	size_t TpB = min((size_t)256, NextMultipleOf(n, 32));
	size_t totalblocks = min((n + TpB - 1) / TpB, (size_t)128);
	dim3 grid = dim3((uint)totalblocks);
	ReKernel<T> <<<grid, (uint)TpB>>> (d_input, d_output, n);
	cudaStreamQuery(0);
}
template void d_Re<tfloat>(tcomplex const* const d_input, tfloat* const d_output, size_t const n);

template <class T> void d_Im(tcomplex const* const d_input, T* const d_output, size_t const n)
{
	size_t TpB = min((size_t)256, NextMultipleOf(n, 32));
	size_t totalblocks = min((n + TpB - 1) / TpB, (size_t)128);
	dim3 grid = dim3((uint)totalblocks);
	ImKernel<T> <<<grid, (uint)TpB>>> (d_input, d_output, n);
}
template void d_Im<tfloat>(tcomplex const* const d_input, tfloat* const d_output, size_t const n);


///////////////////////////////////////
//CUDA kernels for device conversions//
///////////////////////////////////////

template <class Tfrom, class Tto> __global__ void ConvertToKernel(Tfrom const* const d_original, Tto* const d_copy, size_t const n)
{
	for(size_t id = blockIdx.x * blockDim.x + threadIdx.x; 
		id < n; 
		id += blockDim.x * gridDim.x)
		d_copy[id] = (Tto)d_original[id];
}

template <class T> __global__ void ConvertSplitComplexToTComplexKernel(T const* const d_originalr, T const* const d_originali, tcomplex* const d_copy, size_t const n)
{
	for(size_t id = blockIdx.x * blockDim.x + threadIdx.x; 
		id < n; 
		id += blockDim.x * gridDim.x)
	{
		d_copy[id].x = (tfloat)d_originalr[id];
		d_copy[id].y = (tfloat)d_originali[id];
	}
}

template <class T> __global__ void ConvertTComplexToSplitComplexKernel(tcomplex const* const d_original, T* const d_copyr, T* const d_copyi, size_t const n)
{
	for(size_t id = blockIdx.x * blockDim.x + threadIdx.x; 
		id < n; 
		id += blockDim.x * gridDim.x)
	{
		d_copyr[id] = (T)d_original[id].x;
		d_copyi[id] = (T)d_original[id].y;
	}
}

template <class T> __global__ void ReKernel(tcomplex const* const d_input, T* const d_output, size_t const n)
{
	for(size_t id = blockIdx.x * blockDim.x + threadIdx.x; 
		id < n; 
		id += blockDim.x * gridDim.x)
		d_output[id] = (T)d_input[id].x;
}

template <class T> __global__ void ImKernel(tcomplex const* const d_input, T* const d_output, size_t const n)
{
	for(size_t id = blockIdx.x * blockDim.x + threadIdx.x; 
		id < n; 
		id += blockDim.x * gridDim.x)
		d_output[id] = (T)d_input[id].y;
}