#include "Prerequisites.cuh"
#include "Angles.cuh"
#include "DeviceFunctions.cuh"
#include "Helper.cuh"

#define SincWindow 16

////////////////////////////
//CUDA kernel declarations//
////////////////////////////

template<bool iscentered> __global__ void ScaleRotateShift2DSincKernel(tfloat* d_input, tfloat* d_output, int2 dims, glm::mat3 transform);


//////////////////////////////////////////
//Scale, rotate and shift 2D in one step//
//////////////////////////////////////////

void d_ScaleRotateShift2D(tfloat* d_input, tfloat* d_output, int2 dims, tfloat2* h_scales, tfloat* h_angles, tfloat2* h_shifts, T_INTERP_MODE mode, bool outputzerocentered, int batch)
{	
	glm::mat3* h_transforms = (glm::mat3*)malloc(batch * sizeof(glm::mat3));
	for (int b = 0; b < batch; b++)
		h_transforms[b] = Matrix3Translation(tfloat2(-dims.x / 2, -dims.y / 2)) *
						  Matrix3Scale(tfloat3(h_scales[b].x, h_scales[b].y, 1.0f)) *
						  Matrix3RotationZ(-h_angles[b]) *
						  Matrix3Translation(tfloat2(dims.x / 2 - h_shifts[b].x, dims.y / 2 - h_shifts[b].y));

	dim3 TpB = dim3(SincWindow, SincWindow);
	dim3 grid = dim3(dims.x, dims.y);
	for (int b = 0; b < batch; b++)
		if (outputzerocentered)
			ScaleRotateShift2DSincKernel<true> <<<grid, TpB>>> (d_input + dims.x * dims.y * b, d_output + dims.x + dims.y * b, dims, h_transforms[b]);
		else
			ScaleRotateShift2DSincKernel<false> <<<grid, TpB >>> (d_input + dims.x * dims.y * b, d_output + dims.x + dims.y * b, dims, h_transforms[b]);

	free(h_transforms);
}


////////////////
//CUDA kernels//
////////////////

template<bool iscentered> __global__ void ScaleRotateShift2DSincKernel(tfloat* d_input, tfloat* d_output, int2 dims, glm::mat3 transform)
{
	__shared__ tfloat s_sums[SincWindow][SincWindow];
	s_sums[threadIdx.y][threadIdx.x] = 0.0f;

	int outx, outy;
	if (!iscentered)
	{
		outx = dims.x / 2 - blockIdx.x;
		outy = dims.y - 1 - ((blockIdx.y + dims.y / 2 - 1) % dims.y);
	}
	else
	{
		outx = blockIdx.x;
		outy = blockIdx.y;
	}

	glm::vec3 position = glm::vec3(blockIdx.x, blockIdx.y, 1.0f);
	position = transform * position;
	if (position.x < 0 || position.x > dims.x - 1 || position.y < 0 || position.y > dims.y - 1)
	{
		d_output[outy * dims.x + outx] = 0.0f;
		return;
	}

	short startx = (short)position.x - SincWindow / 2;
	short starty = (short)position.y - SincWindow / 2;
	float sum = 0.0f;

	for (short y = threadIdx.y; y < SincWindow; y += blockDim.y)
	{
		short yy = y + starty;
		float weighty = sinc(position.y - (float)yy);
		int addressy = (yy + dims.y) % dims.y;

		for (int x = threadIdx.x; x < SincWindow; x += blockDim.x)
		{
			int xx = x + startx;
			float weight = sinc(position.x - (float)xx) * weighty;
			int addressx = (xx + dims.x) % dims.x;

			sum += d_input[addressy * dims.x + addressx] * weight;
		}
	}
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
		d_output[outy * dims.x + outx] = sum;
	}
}