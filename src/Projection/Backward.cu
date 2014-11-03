#include "Prerequisites.cuh"
#include "Angles.cuh"
#include "DeviceFunctions.cuh"

texture<tfloat, 2> texBackprojImage;

#define SincWindow 16


////////////////////////////
//CUDA kernel declarations//
////////////////////////////

__global__ void ProjBackwardKernel(tfloat* d_volume, int3 dimsvolume, int3 dimsimage, glm::mat4 rotation, float weight);
__global__ void ProjBackwardSincKernel(tfloat* d_volume, int3 dimsvolume, tfloat* d_image, int2 dimsimage, glm::mat4 transform);


/////////////////////////////////////////////
//Equivalent of TOM's tom_backproj3d method//
/////////////////////////////////////////////

void d_ProjBackward(tfloat* d_volume, int3 dimsvolume, tfloat3 offsetfromcenter, tfloat* d_image, int3 dimsimage, tfloat3* h_angles, tfloat2* h_offsets, tfloat2* h_scales, T_INTERP_MODE mode, int batch)
{
	glm::mat4* h_transforms = (glm::mat4*)malloc(batch * sizeof(glm::mat4));
	for (int b = 0; b < batch; b++)
	{
		h_transforms[b] = Matrix4Translation(tfloat3(-dimsvolume.x / 2, -dimsvolume.y / 2, -dimsvolume.z / 2)) *
						  Matrix4Translation(offsetfromcenter) *
						  glm::transpose(Matrix4Euler(h_angles[b])) *
						  Matrix4Scale(tfloat3(1.0f / h_scales[b].x, 1.0f / h_scales[b].y, 1.0f)) *
						  Matrix4Translation(tfloat3(dimsimage.x / 2 - h_offsets[b].x, dimsimage.y / 2 - h_offsets[b].y, 0.0f));
	}

	if (mode == T_INTERP_LINEAR)
	{
		cudaChannelFormatDesc descInput = cudaCreateChannelDesc<tfloat>();
		texBackprojImage.normalized = false;
		texBackprojImage.filterMode = cudaFilterModeLinear;

		size_t TpB = min(192, NextMultipleOf(dimsvolume.x, 32));
		dim3 grid = dim3((dimsvolume.x + TpB - 1) / TpB, dimsvolume.y, dimsvolume.z);
		for (int b = 0; b < batch; b++)
		{
			cudaBindTexture2D(0,
				texBackprojImage,
				d_image + Elements(dimsimage) * b,
				descInput,
				dimsimage.x,
				dimsimage.y,
				dimsimage.x * sizeof(tfloat));


			cudaUnbindTexture(texBackprojImage);
		}
	}
	else if (mode == T_INTERP_SINC)
	{
		dim3 TpB = dim3(SincWindow, SincWindow);
		dim3 grid = dim3(dimsvolume.x, dimsvolume.y, dimsvolume.z);

		for (int b = 0; b < batch; b++)
			ProjBackwardSincKernel <<<grid, TpB>>> (d_volume, dimsvolume, d_image + Elements(dimsimage) * b, toInt2(dimsimage.x, dimsimage.y), h_transforms[b]);
	}

	free(h_transforms);
}


////////////////
//CUDA kernels//
////////////////

__global__ void ProjBackwardKernel(tfloat* d_volume, int3 dimsvolume, int3 dimsimage, glm::mat4 rotation, float weight)
{
	int xvol = blockIdx.x * blockDim.x + threadIdx.x;
	if(xvol >= dimsvolume.x)
		return;

	glm::vec4 voxelpos = glm::vec4((tfloat)(xvol - dimsvolume.x / 2), 
								   (tfloat)((int)blockIdx.y - dimsvolume.y / 2), 
								   (tfloat)((int)blockIdx.z - dimsvolume.z / 2), 
								   1.0f);
	glm::vec4 rotated = rotation * voxelpos;

	rotated.x += (tfloat)(dimsimage.x / 2) + (tfloat)0.5;
	rotated.y += (tfloat)(dimsimage.y / 2) + (tfloat)0.5;
	d_volume[(blockIdx.z * dimsvolume.y + blockIdx.y) * dimsvolume.x + xvol] += weight * tex2D(texBackprojImage, 
																								rotated.x, 
																								rotated.y);
}

template <bool iscentered> __global__ void ProjBackwardSincKernel(tfloat* d_volume, int3 dimsvolume, tfloat* d_image, int2 dimsimage, glm::mat4 transform)
{
	__shared__ tfloat s_sums[SincWindow][SincWindow];
	s_sums[threadIdx.y][threadIdx.x] = 0.0f;

	int outx, outy, outz;
	if (!iscentered)
	{
		outx = gridDim.x / 2 - blockIdx.x;
		outy = gridDim.y - 1 - ((blockIdx.y + gridDim.y / 2 - 1) % gridDim.y);
		outz = gridDim.z - 1 - ((blockIdx.z + gridDim.z / 2 - 1) % gridDim.z);
	}
	else
	{
		outx = blockIdx.x;
		outy = blockIdx.y;
		outz = blockIdx.z;
	}

	glm::vec4 position = glm::vec4(blockIdx.x, blockIdx.y, blockIdx.z, 1.0f);
	position = transform * position;
	if (position.x < 0 || position.x > dimsimage.x - 1 || position.y < 0 || position.y > dimsimage.y - 1)
	{
		d_volume[(outz * gridDim.y + outy) * gridDim.x + outx] = 0.0f;
		return;
	}

	short startx = (short)position.x - SincWindow / 2;
	short starty = (short)position.y - SincWindow / 2;
	float sum = 0.0f;

	for (short y = threadIdx.y; y < SincWindow; y += blockDim.y)
	{
		short yy = y + starty;
		float weighty = sinc(position.y - (float)yy);
		int addressy = (yy + dimsimage.y) % dimsimage.y;

		for (int x = threadIdx.x; x < SincWindow; x += blockDim.x)
		{
			int xx = x + startx;
			float weight = sinc(position.x - (float)xx) * weighty;
			int addressx = (xx + dimsimage.x) % dimsimage.x;

			sum += d_image[addressy * dimsimage.x + addressx] * weight;
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
		d_volume[(outz * gridDim.y + outy) * gridDim.x + outx] = sum;
	}
}