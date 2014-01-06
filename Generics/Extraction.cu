#include "../Prerequisites.cuh"
#include "../Functions.cuh"


////////////////////////////
//CUDA kernel declarations//
////////////////////////////

template <class T> __global__ void ExtractKernel(T* d_input, T* d_output, int3 sourcedims, int3 regiondims, size_t regionelements, int3 regionorigin);


/////////////////////////////////////////////////////////////////////
//Extract a portion of 1/2/3-dimensional data with cyclic boudaries//
/////////////////////////////////////////////////////////////////////

template <class T> void d_Extract(T* d_input, T* d_output, int3 sourcedims, int3 regiondims, int3 regioncenter, int batch)
{
	size_t elementssource = sourcedims.x * sourcedims.y * sourcedims.z;
	size_t elementsregion = regiondims.x * regiondims.y * regiondims.z;

	int3 regionorigin;
	regionorigin.x = (regioncenter.x - (regiondims.x / 2) + sourcedims.x) % sourcedims.x;
	regionorigin.y = (regioncenter.y - (regiondims.y / 2) + sourcedims.y) % sourcedims.y;
	regionorigin.z = (regioncenter.z - (regiondims.z / 2) + sourcedims.z) % sourcedims.z;

	size_t TpB = min(256, NextMultipleOf(regiondims.x, 32));
	dim3 grid = dim3(regiondims.y, regiondims.z, batch);
	ExtractKernel <<<grid, (int)TpB>>> (d_input, d_output, sourcedims, Elements(sourcedims), regiondims, Elements(regiondims), regionorigin);
	cudaStreamQuery(0);
}
template void d_Extract<tfloat>(tfloat* d_input, tfloat* d_output, int3 sourcedims, int3 regiondims, int3 regioncenter, int batch);
template void d_Extract<tcomplex>(tcomplex* d_input, tcomplex* d_output, int3 sourcedims, int3 regiondims, int3 regioncenter, int batch);
template void d_Extract<double>(double* d_input, double* d_output, int3 sourcedims, int3 regiondims, int3 regioncenter, int batch);
template void d_Extract<int>(int* d_input, int* d_output, int3 sourcedims, int3 regiondims, int3 regioncenter, int batch);
template void d_Extract<char>(char* d_input, char* d_output, int3 sourcedims, int3 regiondims, int3 regioncenter, int batch);


////////////////
//CUDA kernels//
////////////////


template <class T> __global__ void ExtractKernel(T* d_input, T* d_output, int3 sourcedims, size_t sourceelements, int3 regiondims, size_t regionelements, int3 regionorigin)
{
	int oy = (blockIdx.x + regionorigin.y) % sourcedims.y;
	int oz = (blockIdx.y + regionorigin.z) % sourcedims.z;

	T* offsetoutput = d_output + blockIdx.z * regionelements + (blockIdx.y * regiondims.y + blockIdx.x) * regiondims.x;
	T* offsetinput = d_input + blockIdx.z * sourceelements + (oz * sourcedims.y + oy) * sourcedims.x;

	for (int x = threadIdx.x; x < regiondims.x; x += blockDim.x)
		offsetoutput[x] = offsetinput[(x + regionorigin.x) % sourcedims.x];
}