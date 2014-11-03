#include "Prerequisites.cuh"
#include "Angles.cuh"
#include "FFT.cuh"
#include "Generics.cuh"
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

__global__ void ReconstructFourierKernel(tcomplex* d_projft, tcomplex* d_volumeft, tfloat* d_samples, int3 dimsvolume, int3 dimsproj, glm::vec3* d_vecX, glm::vec3* d_vecY);

//////////////////////////////////////////////////////
//Performs 3D reconstruction using Fourier inversion//
//////////////////////////////////////////////////////

void d_ReconstructFourierAdd(tcomplex* d_volumeft, tfloat* d_samples, tfloat* d_projections, int3 dimsproj, int3 dimsvolume, tfloat2* h_angles)
{
	tfloat* d_projremapped;
	cudaMalloc((void**)&d_projremapped, Elements(dimsproj) * sizeof(tfloat));
	d_RemapFull2FullFFT(d_projections, d_projremapped, toInt3(dimsproj.x, dimsproj.y, 1), dimsproj.z);

	tcomplex* d_projft;
	cudaMalloc((void**)&d_projft, ElementsFFT(dimsproj) * sizeof(tcomplex));
	for (int b = 0; b < dimsproj.z; b += 100)
		d_FFTR2C(d_projremapped + dimsproj.x * dimsproj.y * b, d_projft + (dimsproj.x / 2 + 1) * dimsproj.y * b, 2, toInt3(dimsproj.x, dimsproj.y, 1), min(100, dimsproj.z - b));
	cudaFree(d_projremapped);

	tcomplex* d_projftshifted;
	cudaMalloc((void**)&d_projftshifted, ElementsFFT(dimsproj) * sizeof(tcomplex));
	d_RemapHalfFFT2Half(d_projft, d_projftshifted, toInt3(dimsproj.x, dimsproj.y, 1), dimsproj.z);
	cudaFree(d_projft);

	glm::vec3* h_vecX = (glm::vec3*)malloc(dimsproj.z * sizeof(glm::vec3));
	glm::vec3* h_vecY = (glm::vec3*)malloc(dimsproj.z * sizeof(glm::vec3));

	for (int b = 0; b < dimsproj.z; b++)
	{
		glm::mat4 mat = Matrix4EulerLegacy(h_angles[b]);
		h_vecX[b] = glm::vec3(mat[0][0], mat[0][1], mat[0][2]);
		h_vecY[b] = glm::vec3(mat[1][0], mat[1][1], mat[1][2]);
	}

	glm::vec3* d_vecX = (glm::vec3*)CudaMallocFromHostArray(h_vecX, dimsproj.z * sizeof(glm::vec3));
	glm::vec3* d_vecY = (glm::vec3*)CudaMallocFromHostArray(h_vecY, dimsproj.z * sizeof(glm::vec3));
	
	free(h_vecX);
	free(h_vecY);

	int TpB = min(NextMultipleOf((dimsproj.x / 2 + 1) * dimsproj.y, 32), 256);
	dim3 grid = dim3(((dimsproj.x / 2 + 1) * dimsproj.y + TpB - 1) / TpB, dimsproj.z);
	ReconstructFourierKernel <<<grid, TpB>>> (d_projftshifted, d_volumeft, d_samples, dimsvolume, toInt3(dimsproj.x, dimsproj.y, 1), d_vecX, d_vecY);

	cudaFree(d_vecX);
	cudaFree(d_vecY);
	cudaFree(d_projftshifted);
}


////////////////
//CUDA kernels//
////////////////

__global__ void ReconstructFourierKernel(tcomplex* d_projft, tcomplex* d_volumeft, tfloat* d_samples, int3 dimsvolume, int3 dimsproj, glm::vec3* d_vecX, glm::vec3* d_vecY)
{
	int elements = ElementsFFT(dimsproj);
	d_projft += elements * blockIdx.y;
	glm::vec3 vecX = d_vecX[blockIdx.y];
	glm::vec3 vecY = d_vecY[blockIdx.y];

	for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < elements; id += gridDim.x * blockDim.x)
	{
		int y = id / (dimsproj.x / 2 + 1);
		int x = id % (dimsproj.x / 2 + 1);

		glm::vec3 rotated = (float)(x - dimsvolume.x / 2) * vecX + (float)(y - dimsvolume.y / 2) * vecY;
		if(rotated.x * rotated.x + rotated.y * rotated.y + rotated.z * rotated.z >= dimsvolume.x * dimsvolume.x / 4)
			continue;

		bool isnegative = false;
		if(rotated.x > 0.0f)
		{
			rotated = -rotated;
			isnegative = true;
		}
		rotated += glm::vec3((float)(dimsvolume.x / 2));
		int x0 = (int)(rotated.x + 0.5f);
		int y0 = (int)(rotated.y + 0.5f);
		int z0 = (int)(rotated.z + 0.5f);

		if(x0 >= dimsvolume.x || y0 >= dimsvolume.y || z0 >= dimsvolume.z)
			continue;

		tcomplex val = d_projft[id];
		if(isnegative)
			val = cconj(val);

		atomicAdd((tfloat*)(d_volumeft + (z0 * dimsvolume.y + y0) * (dimsvolume.x / 2 + 1) + x0), val.x);
		atomicAdd(((tfloat*)(d_volumeft + (z0 * dimsvolume.y + y0) * (dimsvolume.x / 2 + 1) + x0)) + 1, val.y);
		atomicAdd((tfloat*)(d_samples + (z0 * dimsvolume.y + y0) * (dimsvolume.x / 2 + 1) + x0), 1.0f);
	}
}