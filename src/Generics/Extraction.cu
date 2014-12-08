#include "Prerequisites.cuh"
#include "Angles.cuh"
#include "CubicInterp.cuh"
#include "DeviceFunctions.cuh"
#include "Helper.cuh"


////////////////////////////
//CUDA kernel declarations//
////////////////////////////

template <class T> __global__ void ExtractKernel(T* d_input, T* d_output, int3 sourcedims, size_t sourceelements, int3 regiondims, size_t regionelements, int3 regionorigin);
template <class T> __global__ void ExtractManyKernel(T* d_input, T* d_output, int3 sourcedims, size_t sourceelements, int3 regiondims, size_t regionelements, int3* d_regionorigins);
template <bool cubicinterp> __global__ void Extract2DTransformedKernel(cudaTextureObject_t t_input, tfloat* d_output, int2 sourcedims, int2 regiondims, glm::mat3* d_transforms);


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

void d_Extract2DTransformed(tfloat* d_input, tfloat* d_output, int2 dimsinput, int2 dimsregion, tfloat2* h_scale, tfloat* h_rotation, tfloat2* h_translation, T_INTERP_MODE mode, int batch)
{
	cudaArray* a_input;
	cudaTextureObject_t t_input;
	if (mode == T_INTERP_LINEAR)
		d_BindTextureToArray(d_input, a_input, t_input, dimsinput, cudaFilterModeLinear, false);
	else
	{
		tfloat* d_temp;
		cudaMalloc((void**)&d_temp, Elements2(dimsinput) * sizeof(tfloat));
		cudaMemcpy(d_temp, d_input, Elements2(dimsinput) * sizeof(tfloat), cudaMemcpyDeviceToDevice);
		d_CubicBSplinePrefilter2D(d_temp, dimsinput.x * sizeof(tfloat), dimsinput);
		d_BindTextureToArray(d_temp, a_input, t_input, dimsinput, cudaFilterModeLinear, false);
		cudaFree(d_temp);
	}
	
	glm::mat3* h_transforms = (glm::mat3*)malloc(batch * sizeof(glm::mat3));
	for (int b = 0; b < batch; b++)
		h_transforms[b] = Matrix3Translation(tfloat2(h_translation[b].x + dimsregion.x / 2 + 0.5f, h_translation[b].y + dimsregion.y / 2 + 0.5f))*
						  Matrix3Scale(tfloat3(h_scale[b].x, h_scale[b].y, 1.0f)) *
						  glm::transpose(Matrix3RotationZ(h_rotation[b])) *
						  Matrix3Translation(tfloat2(-dimsregion.x / 2, -dimsregion.y / 2));
	glm::mat3* d_transforms = (glm::mat3*)CudaMallocFromHostArray(h_transforms, batch * sizeof(glm::mat3));

	dim3 TpB = dim3(16, 16);
	dim3 grid = dim3((dimsregion.x + 15) / 16, (dimsregion.y + 15) / 16, batch);
	
	if(mode == T_INTERP_MODE::T_INTERP_LINEAR)
		Extract2DTransformedKernel<false> << <grid, TpB >> > (t_input, d_output, dimsinput, dimsregion, d_transforms);
	else if(mode == T_INTERP_MODE::T_INTERP_CUBIC)
		Extract2DTransformedKernel<true> << <grid, TpB >> > (t_input, d_output, dimsinput, dimsregion, d_transforms);

	cudaFree(d_transforms);
	cudaDestroyTextureObject(t_input);
	cudaFreeArray(a_input);
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

template <bool cubicinterp> __global__ void Extract2DTransformedKernel(cudaTextureObject_t t_input, tfloat* d_output, int2 sourcedims, int2 regiondims, glm::mat3* d_transforms)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= regiondims.x)
		return;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	if (idy >= regiondims.y)
		return;
	int idz = blockIdx.z;

	d_output += (idz * regiondims.y + idy) * regiondims.x + idx;

	glm::vec3 pos = d_transforms[idz] * glm::vec3(idx, idy, 1.0f);

	if(pos.x < 0.0f || pos.x > (float)sourcedims.x || pos.y < 0.0f || pos.y > (float)sourcedims.y)
	{
		*d_output = (tfloat)0;
		return;
	}
	else
	{
		if (cubicinterp)
			*d_output = cubicTex2D(t_input, pos.x, pos.y);
		else
			*d_output = tex2D<tfloat>(t_input, pos.x, pos.y);
	}
}