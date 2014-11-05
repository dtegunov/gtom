#include "Prerequisites.cuh"
#include "Angles.cuh"
#include "DeviceFunctions.cuh"
#include "Helper.cuh"


////////////////////////////
//CUDA kernel declarations//
////////////////////////////

template <bool iscentered> __global__ void Exact2DWeightingKernel(tfloat* d_weights, int2 dims, glm::vec3* d_normals, glm::mat3x2* d_globalB2localB, tfloat maxfreq);
template <bool iscentered> __global__ void Exact3DWeightingKernel(tfloat* d_weights, int3 dims, glm::vec3* d_normals, int nimages, tfloat maxfreq);


/////////////////////////////////////////////////////////////////////////////////////
//2D weighting of frequency components for WBP reconstruction, using sinc(distance)//
/////////////////////////////////////////////////////////////////////////////////////

void d_Exact2DWeighting(tfloat* d_weights, int2 dimsimage, tfloat3* h_angles, int nimages, tfloat maxfreq, bool iszerocentered)
{
	glm::vec3* h_normals = (glm::vec3*)malloc(nimages * sizeof(glm::vec3));
	glm::mat3x2* h_globalB2localB = (glm::mat3x2*)malloc(nimages * sizeof(glm::mat3x2));

	for (int i = 0; i < nimages; i++)
	{
		glm::mat3 tB = Matrix3Euler(tfloat3(h_angles[i].x, h_angles[i].y, 0.0f));
		h_normals[i] = glm::vec3(tB[2][0], tB[2][1], tB[2][2]);
		h_globalB2localB[i] = glm::mat3x2(tB[0][0], tB[1][0], tB[0][1], tB[1][1], tB[0][2], tB[1][2]);	//Column-major layout in constructor
	}

	glm::vec3* d_normals = (glm::vec3*)CudaMallocFromHostArray(h_normals, nimages * sizeof(glm::vec3));
	glm::mat3x2* d_globalB2localB = (glm::mat3x2*)CudaMallocFromHostArray(h_globalB2localB, nimages * sizeof(glm::mat3x2));

	uint TpB = min(NextMultipleOf(dimsimage.x / 2 + 1, 32), 128);
	dim3 grid = dim3(dimsimage.y, nimages);
	if (iszerocentered)
		Exact2DWeightingKernel<true> <<<grid, TpB>>> (d_weights, dimsimage, d_normals, d_globalB2localB, maxfreq);
	else
		Exact2DWeightingKernel<false> <<<grid, TpB>>> (d_weights, dimsimage, d_normals, d_globalB2localB, maxfreq);

	free(h_globalB2localB);
	free(h_normals);
	cudaFree(d_globalB2localB);
	cudaFree(d_normals);
}

void d_Exact3DWeighting(tfloat* d_weights, int3 dimsvolume, tfloat3* h_angles, int nimages, tfloat maxfreq, bool iszerocentered)
{
	glm::vec3* h_normals = (glm::vec3*)malloc(nimages * sizeof(glm::vec3));

	for (int i = 0; i < nimages; i++)
	{
		glm::mat3 tB = Matrix3Euler(tfloat3(h_angles[i].x, h_angles[i].y, 0.0f));
		h_normals[i] = glm::vec3(tB[2][0], tB[2][1], tB[2][2]);
	}

	glm::vec3* d_normals = (glm::vec3*)CudaMallocFromHostArray(h_normals, nimages * sizeof(glm::vec3));

	uint TpB = min(NextMultipleOf(dimsvolume.x / 2 + 1, 32), 128);
	dim3 grid = dim3(dimsvolume.y, nimages);
	if (iszerocentered)
		Exact3DWeightingKernel<true> << <grid, TpB >> > (d_weights, dimsvolume, d_normals, nimages, maxfreq);
	else
		Exact3DWeightingKernel<false> << <grid, TpB >> > (d_weights, dimsvolume, d_normals, nimages, maxfreq);

	free(h_normals);
	cudaFree(d_normals);
}


////////////////
//CUDA kernels//
////////////////

template <bool iscentered> __global__ void Exact2DWeightingKernel(tfloat* d_weights, int2 dims, glm::vec3* d_normals, glm::mat3x2* d_globalB2localB, tfloat maxfreq)
{
	int idy = blockIdx.x;
	int interpindex = blockIdx.y;

	int x, y;
	if (!iscentered)
		y = dims.y - 1 - ((idy + dims.y / 2 - 1) % dims.y);
	else
		y = idy;
	d_weights += y * (dims.x / 2 + 1);

	int elements = (dims.x / 2 + 1) * dims.y;
	glm::vec2 center = glm::vec2((float)(dims.x / 2), (float)(dims.y / 2));
	glm::vec3 normalA = d_normals[interpindex];

	for (int idx = threadIdx.x; idx < dims.x / 2 + 1; idx += blockDim.x)
	{
		if (!iscentered)
			x = dims.x / 2 - idx;
		else
			x = idx;

		glm::vec2 localA = glm::vec2((float)idx, (float)idy) - center;
		if (glm::length(localA) <= maxfreq)
		{
			glm::vec3 globalA = localA * d_globalB2localB[interpindex];
			float weightsum = 0.0f;

			for (int b = 0; b < gridDim.y; b++)
			{
				glm::vec3 normalB = d_normals[b];
				float distance = dotp(globalA, normalB);
				weightsum += sinc(distance);
			}

			d_weights[elements * interpindex + x] = 1.0f / weightsum;
		}
		else
		{
			d_weights[elements * interpindex + x] = 0.0f;
		}
	}
}

template <bool iscentered> __global__ void Exact3DWeightingKernel(tfloat* d_weights, int3 dims, glm::vec3* d_normals, int nimages, tfloat maxfreq)
{
	int idy = blockIdx.x;
	int idz = blockIdx.y;

	int x, y, z;
	if (!iscentered)
	{
		y = dims.y - 1 - ((idy + dims.y / 2 - 1) % dims.y);
		z = dims.z - 1 - ((idz + dims.z / 2 - 1) % dims.z);
	}
	else
	{
		y = idy;
		z = idz;
	}
	d_weights += (z * dims.y + y)  * (dims.x / 2 + 1);

	int elements = (dims.x / 2 + 1) * dims.y;
	glm::vec3 center = glm::vec3(dims.x / 2, dims.y / 2, dims.z / 2);

	for (int idx = threadIdx.x; idx < dims.x / 2 + 1; idx += blockDim.x)
	{
		if (!iscentered)
			x = dims.x / 2 - idx;
		else
			x = idx;

		glm::vec3 globalA = glm::vec3(idx, idy, idz) - center;
		if (glm::length(globalA) <= maxfreq)
		{
			float weightsum = 0.0f;

			for (int b = 0; b < nimages; b++)
			{
				glm::vec3 normalB = d_normals[b];
				float distance = dotp(globalA, normalB);
				weightsum += sinc(distance);
			}

			d_weights[x] = 1.0f / max(weightsum, 1.0f);
		}
		else
		{
			d_weights[x] = 0.0f;
		}
	}
}