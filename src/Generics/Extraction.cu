#include "Prerequisites.cuh"
#include "CubicInterp.cuh"
#include "DeviceFunctions.cuh"
#include "Helper.cuh"

#define GLM_FORCE_RADIANS
#define GLM_FORCE_INLINE
#define GLM_FORCE_CUDA
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtx/quaternion.hpp"
#include "glm/gtx/euler_angles.hpp"
#include "glm/gtc/type_ptr.hpp"


////////////////////////////
//CUDA kernel declarations//
////////////////////////////

template <class T> __global__ void ExtractKernel(T* d_input, T* d_output, int3 sourcedims, size_t sourceelements, int3 regiondims, size_t regionelements, int3 regionorigin);
template <class T> __global__ void ExtractManyKernel(T* d_input, T* d_output, int3 sourcedims, size_t sourceelements, int3 regiondims, size_t regionelements, int3* d_regionorigins);
__global__ void Extract2DTransformedLinearKernel(tfloat* d_output, int3 sourcedims, int3 regiondims, glm::vec2* vecx, glm::vec2* vecy, glm::vec2* veccenter);
__global__ void Extract2DTransformedCubicKernel(tfloat* d_output, int3 sourcedims, int3 regiondims, glm::vec2* vecx, glm::vec2* vecy, glm::vec2* veccenter);


///////////
//Globals//
///////////

texture<tfloat, 2, cudaReadModeElementType> texExtractInput2d;


/////////////////////////////////////////////////////////////////////
//Extract a portion of 1/2/3-dimensional data with cyclic boudaries//
/////////////////////////////////////////////////////////////////////

template <class T> void d_Extract(T* d_input, T* d_output, int3 sourcedims, int3 regiondims, int3 regioncenter, int batch)
{
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

template <class T> void d_Extract(T* d_input, T* d_output, int3 sourcedims, int3 regiondims, int3* d_regionorigins, int batch)
{
	size_t TpB = 256;
	dim3 grid = dim3((regiondims.x * regiondims.y + 255) / 256, regiondims.z, batch);
	ExtractManyKernel <<<grid, (int)TpB>>> (d_input, d_output, sourcedims, Elements(sourcedims), regiondims, Elements(regiondims), d_regionorigins);
	cudaStreamQuery(0);
}
template void d_Extract<tfloat>(tfloat* d_input, tfloat* d_output, int3 sourcedims, int3 regiondims, int3* d_regionorigins, int batch);
template void d_Extract<tcomplex>(tcomplex* d_input, tcomplex* d_output, int3 sourcedims, int3 regiondims, int3* d_regionorigins, int batch);
template void d_Extract<double>(double* d_input, double* d_output, int3 sourcedims, int3 regiondims, int3* d_regionorigins, int batch);
template void d_Extract<int>(int* d_input, int* d_output, int3 sourcedims, int3 regiondims, int3* d_regionorigins, int batch);
template void d_Extract<char>(char* d_input, char* d_output, int3 sourcedims, int3 regiondims, int3* d_regionorigins, int batch);


/////////////////////////////////////////////////////////////////////////////////
//Extract a portion of 2-dimensional data with translation and rotation applied//
/////////////////////////////////////////////////////////////////////////////////

void d_Extract2DTransformed(tfloat* d_input, tfloat* d_output, int3 sourcedims, int3 regiondims, tfloat2* h_scale, tfloat* h_rotation, tfloat2* h_translation, T_INTERP_MODE mode, int batch)
{
	texExtractInput2d.normalized = false;
	texExtractInput2d.filterMode = cudaFilterModeLinear;

	int pitchedwidth = sourcedims.x * sizeof(tfloat);
	tfloat* d_pitched = (tfloat*)CudaMallocAligned2D(sourcedims.x * sizeof(tfloat), sourcedims.y, &pitchedwidth);
	for (int y = 0; y < sourcedims.y; y++)
		cudaMemcpy((char*)d_pitched + y * pitchedwidth, 
					d_input + y * sourcedims.x, 
					sourcedims.x * sizeof(tfloat), 
					cudaMemcpyDeviceToDevice);

	if(mode == T_INTERP_CUBIC)
		d_CubicBSplinePrefilter2D(d_pitched, pitchedwidth, toInt2(sourcedims.x, sourcedims.y));

	cudaChannelFormatDesc desc = cudaCreateChannelDesc<tfloat>();
	cudaBindTexture2D(NULL, 
						texExtractInput2d, 
						d_pitched, 
						desc, 
						sourcedims.x, 
						sourcedims.y, 
						pitchedwidth);
	
	glm::vec2* h_vecx = (glm::vec2*)malloc(batch * sizeof(glm::vec2));
	glm::vec2* h_vecy = (glm::vec2*)malloc(batch * sizeof(glm::vec2));
	glm::vec2* h_veccenter = (glm::vec2*)malloc(batch * sizeof(glm::vec2));
	for (int b = 0; b < batch; b++)
	{
		h_vecx[b] = glm::vec2(cos(h_rotation[b]), sin(h_rotation[b])) * h_scale[b].x;
		h_vecy[b] = glm::vec2(cos(h_rotation[b] + PI / (tfloat)2), sin(h_rotation[b] + PI / (tfloat)2)) * h_scale[b].y;
		h_veccenter[b] = glm::vec2(h_translation[b].x, h_translation[b].y);
	}

	glm::vec2* d_vecx = (glm::vec2*)CudaMallocFromHostArray(h_vecx, batch * sizeof(glm::vec2));
	glm::vec2* d_vecy = (glm::vec2*)CudaMallocFromHostArray(h_vecy, batch * sizeof(glm::vec2));
	glm::vec2* d_veccenter = (glm::vec2*)CudaMallocFromHostArray(h_veccenter, batch * sizeof(glm::vec2));

	free(h_vecx);
	free(h_vecy);
	free(h_veccenter);

	size_t TpB = min(256, NextMultipleOf(regiondims.x, 32));
	dim3 grid = dim3((regiondims.x + TpB - 1) / TpB, regiondims.y, batch);
	
	if(mode == T_INTERP_MODE::T_INTERP_LINEAR)
		Extract2DTransformedLinearKernel <<<grid, (int)TpB>>> (d_output, sourcedims, regiondims, d_vecx, d_vecy, d_veccenter);
	else if(mode == T_INTERP_MODE::T_INTERP_CUBIC)
		Extract2DTransformedCubicKernel <<<grid, (int)TpB>>> (d_output, sourcedims, regiondims, d_vecx, d_vecy, d_veccenter);

	cudaStreamQuery(0);

	cudaUnbindTexture(texExtractInput2d);
	cudaFree(d_vecx);
	cudaFree(d_vecy);
	cudaFree(d_veccenter);
	cudaFree(d_pitched);
}

glm::mat4 GetTransform2D(tfloat2 scale, tfloat rotation, tfloat2 translation)
{
	return glm::translate(glm::rotate(glm::scale(glm::mat4(1.0f), glm::vec3(scale.x, scale.y, 1.0f)), glm::radians(rotation), glm::vec3(0.0f, 0.0f, 1.0f)), glm::vec3(translation.x, translation.y, 0.0f));
}


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

template <class T> __global__ void ExtractManyKernel(T* d_input, T* d_output, int3 sourcedims, size_t sourceelements, int3 regiondims, size_t regionelements, int3* d_regionorigins)
{
	int3 regionorigin = d_regionorigins[blockIdx.z];
	uint id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id >= regiondims.x * regiondims.y)
		return;
	
	uint y = id / regiondims.x;
	uint x = id - y * regiondims.x;
	int oy = (y + regionorigin.y);
	int oz = (blockIdx.y + regionorigin.z);

	d_output += blockIdx.z * regionelements + (blockIdx.y * regiondims.y + y) * regiondims.x;
	d_input += (oz * sourcedims.y + oy) * sourcedims.x + regionorigin.x;

	d_output[x] = d_input[x];
}

__global__ void Extract2DTransformedLinearKernel(tfloat* d_output, int3 sourcedims, int3 regiondims, glm::vec2* vecx, glm::vec2* vecy, glm::vec2* veccenter)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= regiondims.x)
		return;
	int idy = blockIdx.y;

	d_output += blockIdx.z * Elements(regiondims) + getOffset(idx, idy, regiondims.x);

	glm::vec2 pos = veccenter[blockIdx.z] + (vecx[blockIdx.z] * (float)(idx - regiondims.x / 2)) + (vecy[blockIdx.z] * (float)(idy - regiondims.y / 2)) + glm::vec2(0.5f, 0.5f);

	if(pos.x < 0.0f || pos.x > (float)sourcedims.x || pos.y < 0.0f || pos.y > (float)sourcedims.y)
	{
		*d_output = (tfloat)0;
		return;
	}
	else
	{
		*d_output = tex2D(texExtractInput2d, pos.x, pos.y);
	}
}

__global__ void Extract2DTransformedCubicKernel(tfloat* d_output, int3 sourcedims, int3 regiondims, glm::vec2* vecx, glm::vec2* vecy, glm::vec2* veccenter)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= regiondims.x)
		return;
	int idy = blockIdx.y;

	d_output += blockIdx.z * Elements(regiondims) + getOffset(idx, idy, regiondims.x);

	glm::vec2 pos = veccenter[blockIdx.z] + (vecx[blockIdx.z] * (float)(idx - regiondims.x / 2)) + (vecy[blockIdx.z] * (float)(idy - regiondims.y / 2)) + glm::vec2(0.5f, 0.5f);

	if(pos.x < 0.0f || pos.x > (float)sourcedims.x || pos.y < 0.0f || pos.y > (float)sourcedims.y)
	{
		*d_output = (tfloat)0;
		return;
	}
	else
	{
		*d_output = cubicTex2D(texExtractInput2d, pos.x, pos.y);
	}
}