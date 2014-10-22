#include "Prerequisites.cuh"
#include "Angles.cuh"
#include "FFT.cuh"
#include "Generics.cuh"
#include "Helper.cuh"
#include "Reconstruction.cuh"

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

//__global__ void ReconstructFourierKernel(tcomplex* d_projft, tcomplex* d_volumeft, tfloat* d_samples, int3 dimsvolume, int3 dimsproj, glm::vec3* d_vecX, glm::vec3* d_vecY);

//////////////////////////////////////////////////////
//Performs 3D reconstruction using Fourier inversion//
//////////////////////////////////////////////////////

void d_ReconstructFourier(tfloat* d_projections, int3 dimsproj, tfloat* d_volume, int3 dimsvolume, tfloat2* h_angles)
{
	tcomplex* d_volumeft = (tcomplex*)CudaMallocValueFilled(ElementsFFT(dimsvolume) * 2, (tfloat)0);
	tfloat* d_samples = CudaMallocValueFilled(ElementsFFT(dimsvolume), (tfloat)0);

	d_ReconstructFourierAdd(d_volumeft, d_samples, d_projections, dimsproj, dimsvolume, h_angles);

	d_Inv(d_samples, d_samples, ElementsFFT(dimsvolume));
	d_ComplexMultiplyByVector(d_volumeft, d_samples, d_volumeft, ElementsFFT(dimsvolume));
	cudaFree(d_samples);
	d_RemapHalf2HalfFFT(d_volumeft, d_volumeft, dimsvolume);

	d_IFFTC2R(d_volumeft, d_volume, 3, dimsvolume);
	d_RemapFullFFT2Full(d_volume, d_volume, dimsvolume);


	cudaFree(d_volumeft);
}


////////////////
//CUDA kernels//
////////////////

/*__global__ void ReconstructFourierKernel(tcomplex* d_projft, tcomplex* d_volumeft, tfloat* d_samples, int3 dimsvolume, int3 dimsproj, glm::vec3* d_vecX, glm::vec3* d_vecY)
{
	int elements = ElementsFFT(dimsproj);
	d_projft += elements * blockIdx.y;

	for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < elements; id += gridDim.x * blockDim.x)
	{
		int y = id / (dimsproj.x / 2 + 1);
		int x = id % (dimsproj.x / 2 + 1);

		glm::vec3 rotated = (float)(x - dimsvolume.x / 2) * d_vecX[blockIdx.y] + (float)(y - dimsvolume.y / 2) * d_vecY[blockIdx.y];
		if(rotated.x * rotated.x + rotated.y * rotated.y + rotated.z * rotated.z >= dimsvolume.x * dimsvolume.x / 4)
			continue;

		bool isnegative = false;
		if(rotated.x > 0.0f)
		{
			rotated = -rotated;
			isnegative = true;
		}
		rotated += glm::vec3((float)(dimsvolume.x / 2));
		int x0 = (int)rotated.x;
		int y0 = (int)rotated.y;
		int z0 = (int)rotated.z;

		if(x0 >= dimsvolume.x || y0 >= dimsvolume.y || z0 >= dimsvolume.z)
			continue;

		int x1 = min(x0 + 1, dimsvolume.x - 1);
		int y1 = min(y0 + 1, dimsvolume.y - 1);
		int z1 = min(z0 + 1, dimsvolume.z - 1);

		tcomplex val = d_projft[id];
		if(isnegative)
			val = cconj(val);
		
		float xd = rotated.x - floor(rotated.x);
		float yd = rotated.y - floor(rotated.y);
		float zd = rotated.z - floor(rotated.z);

		float c0 = 1.0f - zd;
		float c1 = zd;

		float c00 = (1.0f - yd) * c0;
		float c10 = yd * c0;
		float c01 = (1.0f - yd) * c1;
		float c11 = yd * c1;

		float c000 = (1.0f - xd) * c00;
		float c100 = xd * c00;
		float c010 = (1.0f - xd) * c10;
		float c110 = xd * c10;
		float c001 = (1.0f - xd) * c01;
		float c101 = xd * c01;
		float c011 = (1.0f - xd) * c11;
		float c111 = xd * c11;

		atomicAdd((tfloat*)(d_volumeft + (z0 * dimsvolume.y + y0) * (dimsvolume.x / 2 + 1) + x0), c000 * val.x);
		atomicAdd(((tfloat*)(d_volumeft + (z0 * dimsvolume.y + y0) * (dimsvolume.x / 2 + 1) + x0)) + 1, c000 * val.y);
		atomicAdd((tfloat*)(d_samples + (z0 * dimsvolume.y + y0) * (dimsvolume.x / 2 + 1) + x0), c000);

		atomicAdd((tfloat*)(d_volumeft + (z0 * dimsvolume.y + y0) * (dimsvolume.x / 2 + 1) + x1), c100 * val.x);
		atomicAdd(((tfloat*)(d_volumeft + (z0 * dimsvolume.y + y0) * (dimsvolume.x / 2 + 1) + x1)) + 1, c100 * val.y);
		atomicAdd((tfloat*)(d_samples + (z0 * dimsvolume.y + y0) * (dimsvolume.x / 2 + 1) + x1), c100);

		atomicAdd((tfloat*)(d_volumeft + (z0 * dimsvolume.y + y1) * (dimsvolume.x / 2 + 1) + x0), c010 * val.x);
		atomicAdd(((tfloat*)(d_volumeft + (z0 * dimsvolume.y + y1) * (dimsvolume.x / 2 + 1) + x0)) + 1, c010 * val.y);
		atomicAdd((tfloat*)(d_samples + (z0 * dimsvolume.y + y1) * (dimsvolume.x / 2 + 1) + x0), c010);

		atomicAdd((tfloat*)(d_volumeft + (z0 * dimsvolume.y + y1) * (dimsvolume.x / 2 + 1) + x1), c110 * val.x);
		atomicAdd(((tfloat*)(d_volumeft + (z0 * dimsvolume.y + y1) * (dimsvolume.x / 2 + 1) + x1)) + 1, c110 * val.y);
		atomicAdd((tfloat*)(d_samples + (z0 * dimsvolume.y + y1) * (dimsvolume.x / 2 + 1) + x1), c110);


		atomicAdd((tfloat*)(d_volumeft + (z1 * dimsvolume.y + y0) * (dimsvolume.x / 2 + 1) + x0), c001 * val.x);
		atomicAdd(((tfloat*)(d_volumeft + (z1 * dimsvolume.y + y0) * (dimsvolume.x / 2 + 1) + x0)) + 1, c001 * val.y);
		atomicAdd((tfloat*)(d_samples + (z1 * dimsvolume.y + y0) * (dimsvolume.x / 2 + 1) + x0), c001);

		atomicAdd((tfloat*)(d_volumeft + (z1 * dimsvolume.y + y0) * (dimsvolume.x / 2 + 1) + x1), c101 * val.x);
		atomicAdd(((tfloat*)(d_volumeft + (z1 * dimsvolume.y + y0) * (dimsvolume.x / 2 + 1) + x1)) + 1, c101 * val.y);
		atomicAdd((tfloat*)(d_samples + (z1 * dimsvolume.y + y0) * (dimsvolume.x / 2 + 1) + x1), c101);

		atomicAdd((tfloat*)(d_volumeft + (z1 * dimsvolume.y + y1) * (dimsvolume.x / 2 + 1) + x0), c011 * val.x);
		atomicAdd(((tfloat*)(d_volumeft + (z1 * dimsvolume.y + y1) * (dimsvolume.x / 2 + 1) + x0)) + 1, c011 * val.y);
		atomicAdd((tfloat*)(d_samples + (z1 * dimsvolume.y + y1) * (dimsvolume.x / 2 + 1) + x0), c011);

		atomicAdd((tfloat*)(d_volumeft + (z1 * dimsvolume.y + y1) * (dimsvolume.x / 2 + 1) + x1), c111 * val.x);
		atomicAdd(((tfloat*)(d_volumeft + (z1 * dimsvolume.y + y1) * (dimsvolume.x / 2 + 1) + x1)) + 1, c111 * val.y);
		atomicAdd((tfloat*)(d_samples + (z1 * dimsvolume.y + y1) * (dimsvolume.x / 2 + 1) + x1), c111);
	}
}*/