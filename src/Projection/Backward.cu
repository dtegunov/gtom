#include "Prerequisites.cuh"
#include "Angles.cuh"
#include "CubicInterp.cuh"
#include "DeviceFunctions.cuh"
#include "Helper.cuh"

#define SincWindow 16


////////////////////////////
//CUDA kernel declarations//
////////////////////////////

template <bool iscentered, bool cubicinterp> __global__ void ProjBackwardKernel(tfloat* d_volume, int3 dimsvolume, cudaTextureObject_t t_image, int2 dimsimage, glm::mat4 transform);
template <bool iscentered> __global__ void ProjBackwardSincKernel(tfloat* d_volume, int3 dimsvolume, tfloat* d_image, int2 dimsimage, glm::mat4 transform);


/////////////////////////////////////////////
//Equivalent of TOM's tom_backproj3d method//
/////////////////////////////////////////////

void d_ProjBackward(tfloat* d_volume, int3 dimsvolume, tfloat3 offsetfromcenter, tfloat* d_image, int3 dimsimage, tfloat3* h_angles, tfloat2* h_offsets, tfloat2* h_scales, T_INTERP_MODE mode, bool outputzerocentered, int batch)
{
	glm::mat4* h_transforms = (glm::mat4*)malloc(batch * sizeof(glm::mat4));
	for (int b = 0; b < batch; b++)
	{
		h_transforms[b] = Matrix4Translation(tfloat3((tfloat)dimsimage.x / 2.0f + 0.5f, (tfloat)dimsimage.y / 2.0f + 0.5f, 0.0f)) *
						  Matrix4Scale(tfloat3(1.0f / h_scales[b].x, 1.0f / h_scales[b].y, 1.0f)) *
						  Matrix4Translation(tfloat3(-h_offsets[b].x, -h_offsets[b].y, 0.0f)) *
						  glm::transpose(Matrix4Euler(h_angles[b])) *
						  Matrix4Translation(offsetfromcenter) *
						  Matrix4Translation(tfloat3(-dimsvolume.x / 2, -dimsvolume.y / 2, -dimsvolume.z / 2));
	}

	if (mode == T_INTERP_LINEAR || mode == T_INTERP_CUBIC)
	{
		cudaArray* a_image;
		cudaTextureObject_t t_image;
		tfloat* d_temp;
		cudaMalloc((void**)&d_temp, Elements(dimsimage) * sizeof(tfloat));

		for (int b = 0; b < batch; b++)
		{
			cudaMemcpy(d_temp, d_image + Elements(dimsimage) * b, Elements(dimsimage) * sizeof(tfloat), cudaMemcpyDeviceToDevice);
			if (mode == T_INTERP_CUBIC)
				d_CubicBSplinePrefilter2D(d_temp, dimsimage.x * sizeof(tfloat), toInt2(dimsimage.x, dimsimage.y));
			d_BindTextureToArray(d_temp, a_image, t_image, toInt2(dimsimage.x, dimsimage.y), cudaFilterModeLinear, false);

			dim3 TpB = dim3(16, 16);
			dim3 grid = dim3((dimsvolume.x + 15) / 16, (dimsvolume.y + 15) / 16, dimsvolume.z);
			
			if (outputzerocentered)
			{
				if (mode == T_INTERP_LINEAR)
					ProjBackwardKernel<true, false> << <grid, TpB >> > (d_volume, dimsvolume, t_image, toInt2(dimsimage.x, dimsimage.y), h_transforms[b]);
				else
					ProjBackwardKernel<true, true> << <grid, TpB >> > (d_volume, dimsvolume, t_image, toInt2(dimsimage.x, dimsimage.y), h_transforms[b]);
			}
			else
			{
				if (mode == T_INTERP_LINEAR)
					ProjBackwardKernel<false, false> << <grid, TpB >> > (d_volume, dimsvolume, t_image, toInt2(dimsimage.x, dimsimage.y), h_transforms[b]);
				else
					ProjBackwardKernel<false, true> << <grid, TpB >> > (d_volume, dimsvolume, t_image, toInt2(dimsimage.x, dimsimage.y), h_transforms[b]);
			}

			cudaDestroyTextureObject(t_image);
			cudaFreeArray(a_image);
		}

		cudaFree(d_temp);
	}
	else if (mode == T_INTERP_SINC)
	{
		dim3 TpB = dim3(SincWindow, SincWindow);
		dim3 grid = dim3(dimsvolume.x, dimsvolume.y, dimsvolume.z);

		for (int b = 0; b < batch; b++)
			if (outputzerocentered)
				ProjBackwardSincKernel<true> <<<grid, TpB >>> (d_volume, dimsvolume, d_image + Elements(dimsimage) * b, toInt2(dimsimage.x, dimsimage.y), h_transforms[b]);
			else
				ProjBackwardSincKernel<false> <<<grid, TpB >>> (d_volume, dimsvolume, d_image + Elements(dimsimage) * b, toInt2(dimsimage.x, dimsimage.y), h_transforms[b]);
	}

	free(h_transforms);
}


////////////////
//CUDA kernels//
////////////////

template <bool iscentered, bool cubicinterp> __global__ void ProjBackwardKernel(tfloat* d_volume, int3 dimsvolume, cudaTextureObject_t t_image, int2 dimsimage, glm::mat4 transform)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= dimsvolume.x)
		return;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	if (idy >= dimsvolume.y)
		return;
	int idz = blockIdx.z;

	int outx, outy, outz;
	if (!iscentered)
	{
		outx = (idx + (dimsvolume.x + 1) / 2) % dimsvolume.x;
		outy = (idy + (dimsvolume.y + 1) / 2) % dimsvolume.y;
		outz = (idz + (dimsvolume.z + 1) / 2) % dimsvolume.z;
	}
	else
	{
		outx = idx;
		outy = idy;
		outz = idz;
	}

	glm::vec4 position = glm::vec4(idx, idy, idz, 1);
	position = transform * position;

	if (cubicinterp)
		d_volume[(outz * dimsvolume.y + outy) * dimsvolume.x + outx] += cubicTex2D(t_image, position.x, position.y);
	else
		d_volume[(outz * dimsvolume.y + outy) * dimsvolume.x + outx] += tex2D<tfloat>(t_image, position.x, position.y);
}

template <bool iscentered> __global__ void ProjBackwardSincKernel(tfloat* d_volume, int3 dimsvolume, tfloat* d_image, int2 dimsimage, glm::mat4 transform)
{
	__shared__ float s_sums[SincWindow][SincWindow];
	s_sums[threadIdx.y][threadIdx.x] = 0.0f;

	int outx, outy, outz;
	if (!iscentered)
	{
		outx = (blockIdx.x + (dimsvolume.x + 1) / 2) % dimsvolume.x;
		outy = (blockIdx.y + (dimsvolume.y + 1) / 2) % dimsvolume.y;
		outz = (blockIdx.z + (dimsvolume.z + 1) / 2) % dimsvolume.z;
	}
	else
	{
		outx = blockIdx.x;
		outy = blockIdx.y;
		outz = blockIdx.z;
	}

	glm::vec4 position = glm::vec4((int)blockIdx.x, (int)blockIdx.y, (int)blockIdx.z, 1);
	position = transform * position;
	if (position.x < 0 || position.x > dimsimage.x - 1 || position.y < 0 || position.y > dimsimage.y - 1)
		return;

	int startx = (int)position.x - SincWindow / 2;
	int starty = (int)position.y - SincWindow / 2;
	float sum = 0.0;

	int y = (int)threadIdx.y + starty;
	int addressy = (y + dimsimage.y) % dimsimage.y;

	int x = (int)threadIdx.x + startx;
	float weight = sinc(position.x - (float)x) * sinc(position.y - (float)y);
	int addressx = (x + dimsimage.x) % dimsimage.x;

	sum += d_image[addressy * dimsimage.x + addressx] * weight;

	s_sums[threadIdx.y][threadIdx.x] = sum;
	__syncthreads();
	
	if (threadIdx.x == 0)
	{
		#pragma unroll
		for (char i = 1; i < SincWindow; i++)
			sum += s_sums[threadIdx.y][i];
		s_sums[threadIdx.y][0] = sum;
	}
	__syncthreads();

	if (threadIdx.y == 0 && threadIdx.x == 0)
	{
		#pragma unroll
		for (char i = 1; i < SincWindow; i++)
			sum += s_sums[i][0];
		d_volume[(outz * gridDim.y + outy) * gridDim.x + outx] += sum;
	}
}