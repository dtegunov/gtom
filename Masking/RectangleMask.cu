#include "..\Prerequisites.cuh"
#include "..\Functions.cuh"


////////////////////////////
//CUDA kernel declarations//
////////////////////////////

template <class T> __global__ void RectangleMaskKernel(T const* const d_input, T* const d_output, int3 const size, int2 const limx, int2 const limy, int2 const limz);


////////////////
//Host methods//
////////////////

template <class T> void d_RectangleMask(T const* const d_input, 
										T* const d_output, 
										int3 const size,
										int3 const rectsize,
										tfloat const sigma,
										int3 const* const center,
										int batch)
{
	size_t elements = size.x * size.y * size.z;
	size_t elementsFFT = (size.x / 2 + 1) * size.y * size.z;

	int3 _center = center != NULL ? *center : toInt3(size.x / 2, size.y / 2, size.z / 2);

	int2 limx = toInt2(_center.x - rectsize.x / 2, _center.x + rectsize.x / 2);
	int2 limy = toInt2(_center.y - rectsize.y / 2, _center.y + rectsize.y / 2);
	int2 limz = toInt2(_center.z - rectsize.z / 2, _center.z + rectsize.z / 2);

	int TpB = min(256, size.x);
	dim3 grid = dim3(size.y, size.z, batch);
	RectangleMaskKernel<T> <<<grid, TpB>>> (d_input, d_output, size, limx, limy, limz);

	if(sigma > 0)
	{
		tfloat* d_mask = CudaMallocValueFilled(elements, (tfloat)1);
		d_SphereMask(d_mask, d_mask, size, &sigma, (tfloat)0, (tfloat3*)NULL);
		tfloat* d_summask;
		cudaMalloc((void**)&d_summask, sizeof(tfloat));
		d_Sum(d_mask, d_summask, elements);
		tfloat* h_summask = (tfloat*)MallocFromDeviceArray(d_summask, sizeof(tfloat));
		cudaFree(d_summask);

		tcomplex* d_maskFFT;
		cudaMalloc((void**)&d_maskFFT, elementsFFT * sizeof(tcomplex));
		d_FFTR2C(d_mask, d_maskFFT, NumOfDims(size), size);
		cudaFree(d_mask);

		tcomplex* d_outputFFT;
		cudaMalloc((void**)&d_outputFFT, elementsFFT * sizeof(tcomplex));
		d_FFTR2C(d_output, d_outputFFT, NumOfDims(size), size);

		d_ComplexMultiplyByVector(d_outputFFT, d_maskFFT, d_outputFFT, elementsFFT);
		cudaFree(d_maskFFT);

		tfloat* d_intermediate;
		cudaMalloc((void**)&d_intermediate, elements * sizeof(tfloat));
		d_IFFTC2R(d_outputFFT, d_intermediate, NumOfDims(size), size);
		cudaFree(d_outputFFT);

		d_RemapFullFFT2Full(d_intermediate, d_output, size);
		d_MultiplyByScalar(d_output, d_output, elements, (tfloat)1 / h_summask[0]);
		cudaFree(d_intermediate);
	}

	cudaDeviceSynchronize();
}
template void d_RectangleMask<tfloat>(tfloat const* const d_input, tfloat* const d_output, int3 const size, int3 const rectsize, tfloat const sigma, int3 const* const center, int batch);


////////////////
//CUDA kernels//
////////////////

template <class T> __global__ void RectangleMaskKernel(T const* const d_input, T* const d_output, int3 const size, int2 const limx, int2 const limy, int2 const limz)
{
	if(threadIdx.x >= size.x)
		return;
	//For batch mode
	int offset = blockIdx.z * size.x * size.y * size.z + blockIdx.y * size.x * size.y + blockIdx.x * size.x;

	if(blockIdx.x >= limy.x && blockIdx.x <= limy.y && blockIdx.y >= limz.x && blockIdx.y <= limz.y)
		for(int x = threadIdx.x; x < size.x; x += blockDim.x)
			if(x >= limx.x && x <= limx.y)
				d_output[offset + x] = (T)1;
			else
				d_output[offset + x] = d_input[offset + x];
	else
		for(int x = threadIdx.x; x < size.x; x += blockDim.x)
			d_output[offset + x] = d_input[offset + x];
}