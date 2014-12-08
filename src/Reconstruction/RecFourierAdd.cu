#include "Prerequisites.cuh"
#include "Angles.cuh"
#include "CTF.cuh"
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

template<bool weighted, bool ctfed> __global__ void ReconstructFourierKernel(tcomplex* d_projft, tfloat* d_weights, CTFParamsLean* d_ctf, tcomplex* d_volumeft, tfloat* d_samples, int3 dimsvolume, int3 dimsproj, glm::vec3* d_vecX, glm::vec3* d_vecY);

//////////////////////////////////////////////////////
//Performs 3D reconstruction using Fourier inversion//
//////////////////////////////////////////////////////

void d_ReconstructFourierAdd(tcomplex* d_volumeft, tfloat* d_samples, tfloat* d_projections, tfloat* d_weights, CTFParams* h_ctf, int3 dimsproj, int3 dimsvolume, tfloat3* h_angles)
{
	tfloat* d_projremapped;
	cudaMalloc((void**)&d_projremapped, Elements(dimsproj) * sizeof(tfloat));
	d_RemapFull2FullFFT(d_projections, d_projremapped, toInt3(dimsproj.x, dimsproj.y, 1), dimsproj.z);

	tcomplex* d_projft;
	cudaMalloc((void**)&d_projft, ElementsFFT(dimsproj) * sizeof(tcomplex));
	for (int b = 0; b < dimsproj.z; b += 100)
		d_FFTR2C(d_projremapped + dimsproj.x * dimsproj.y * b, d_projft + (dimsproj.x / 2 + 1) * dimsproj.y * b, 2, toInt3(dimsproj.x, dimsproj.y, 1), min(100, dimsproj.z - b));
	cudaFree(d_projremapped);

	CTFParamsLean* d_ctf = NULL;
	if (d_weights == NULL && h_ctf != NULL)
	{
		CTFParamsLean* h_ctflean = (CTFParamsLean*)malloc(dimsproj.z * sizeof(CTFParamsLean));
		for (int n = 0; n < dimsproj.z; n++)
			h_ctflean[n] = CTFParamsLean(h_ctf[n]);
		d_ctf = (CTFParamsLean*)CudaMallocFromHostArray(h_ctflean, dimsproj.z * sizeof(CTFParamsLean));
	}

	glm::vec3* h_vecX = (glm::vec3*)malloc(dimsproj.z * sizeof(glm::vec3));
	glm::vec3* h_vecY = (glm::vec3*)malloc(dimsproj.z * sizeof(glm::vec3));

	for (int b = 0; b < dimsproj.z; b++)
	{
		glm::mat3 mat = Matrix3Euler(h_angles[b]);
		h_vecX[b] = glm::vec3(mat[0][0], mat[0][1], mat[0][2]);
		h_vecY[b] = glm::vec3(mat[1][0], mat[1][1], mat[1][2]);
	}

	glm::vec3* d_vecX = (glm::vec3*)CudaMallocFromHostArray(h_vecX, dimsproj.z * sizeof(glm::vec3));
	glm::vec3* d_vecY = (glm::vec3*)CudaMallocFromHostArray(h_vecY, dimsproj.z * sizeof(glm::vec3));	
	free(h_vecX);
	free(h_vecY);

	int TpB = min(NextMultipleOf(ElementsFFT2(dimsproj), 32), 256);
	dim3 grid = dim3((ElementsFFT2(dimsproj) + TpB - 1) / TpB, dimsproj.z);
	if (d_weights != NULL)
		ReconstructFourierKernel<true, false> << <grid, TpB >> > (d_projft, d_weights, d_ctf, d_volumeft, d_samples, dimsvolume, toInt3(dimsproj.x, dimsproj.y, 1), d_vecX, d_vecY);
	else if (d_ctf != NULL)
		ReconstructFourierKernel<false, true> << <grid, TpB >> > (d_projft, d_weights, d_ctf, d_volumeft, d_samples, dimsvolume, toInt3(dimsproj.x, dimsproj.y, 1), d_vecX, d_vecY);
	else
		ReconstructFourierKernel<false, false> << <grid, TpB >> > (d_projft, d_weights, d_ctf, d_volumeft, d_samples, dimsvolume, toInt3(dimsproj.x, dimsproj.y, 1), d_vecX, d_vecY);

	cudaFree(d_vecX);
	cudaFree(d_vecY);
	cudaFree(d_projft);
}


////////////////
//CUDA kernels//
////////////////

template<bool weighted, bool ctfed> __global__ void ReconstructFourierKernel(tcomplex* d_projft, tfloat* d_weights, CTFParamsLean* d_ctf, tcomplex* d_volumeft, tfloat* d_samples, int3 dimsvolume, int3 dimsproj, glm::vec3* d_vecX, glm::vec3* d_vecY)
{
	int elementsrow = ElementsFFT1(dimsproj.x);
	int elements = ElementsFFT(dimsproj);
	d_projft += elements * blockIdx.y;
	if (weighted)
		d_weights += elements * blockIdx.y;
	if (ctfed)
		d_ctf += blockIdx.y;

	glm::vec3 vecX = d_vecX[blockIdx.y];
	glm::vec3 vecY = d_vecY[blockIdx.y];

	for (uint id = blockIdx.x * blockDim.x + threadIdx.x; id < elements; id += gridDim.x * blockDim.x)
	{
		int x = id % elementsrow;
		x = -x;
		int y = id / elementsrow;
		y = dimsproj.y / 2 - 1 - ((y + dimsproj.y / 2 - 1) % dimsproj.y);

		tfloat weight = weighted ? d_weights[id] : (tfloat)1;
		if (ctfed)
		{
			CTFParamsLean ctf = *d_ctf;
			double k = sqrt((double)(x * x + y * y)) * ctf.ny * 2.0 / (double)dimsproj.x;
			double angle = atan2((double)y, (double)x);
			weight = d_GetCTF<false>(k, angle, ctf);
		}

		glm::vec3 rotated = (float)x * vecX + (float)y * vecY;
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
		if (ctfed && weight < 0)
		{
			val = make_cuComplex(-val.x, -val.y);
			weight = abs(weight);
		}

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
		atomicAdd((tfloat*)(d_samples + (z0 * dimsvolume.y + y0) * (dimsvolume.x / 2 + 1) + x0), c000 * weight);

		atomicAdd((tfloat*)(d_volumeft + (z0 * dimsvolume.y + y0) * (dimsvolume.x / 2 + 1) + x1), c100 * val.x);
		atomicAdd(((tfloat*)(d_volumeft + (z0 * dimsvolume.y + y0) * (dimsvolume.x / 2 + 1) + x1)) + 1, c100 * val.y);
		atomicAdd((tfloat*)(d_samples + (z0 * dimsvolume.y + y0) * (dimsvolume.x / 2 + 1) + x1), c100 * weight);

		atomicAdd((tfloat*)(d_volumeft + (z0 * dimsvolume.y + y1) * (dimsvolume.x / 2 + 1) + x0), c010 * val.x);
		atomicAdd(((tfloat*)(d_volumeft + (z0 * dimsvolume.y + y1) * (dimsvolume.x / 2 + 1) + x0)) + 1, c010 * val.y);
		atomicAdd((tfloat*)(d_samples + (z0 * dimsvolume.y + y1) * (dimsvolume.x / 2 + 1) + x0), c010 * weight);

		atomicAdd((tfloat*)(d_volumeft + (z0 * dimsvolume.y + y1) * (dimsvolume.x / 2 + 1) + x1), c110 * val.x);
		atomicAdd(((tfloat*)(d_volumeft + (z0 * dimsvolume.y + y1) * (dimsvolume.x / 2 + 1) + x1)) + 1, c110 * val.y);
		atomicAdd((tfloat*)(d_samples + (z0 * dimsvolume.y + y1) * (dimsvolume.x / 2 + 1) + x1), c110 * weight);


		atomicAdd((tfloat*)(d_volumeft + (z1 * dimsvolume.y + y0) * (dimsvolume.x / 2 + 1) + x0), c001 * val.x);
		atomicAdd(((tfloat*)(d_volumeft + (z1 * dimsvolume.y + y0) * (dimsvolume.x / 2 + 1) + x0)) + 1, c001 * val.y);
		atomicAdd((tfloat*)(d_samples + (z1 * dimsvolume.y + y0) * (dimsvolume.x / 2 + 1) + x0), c001 * weight);

		atomicAdd((tfloat*)(d_volumeft + (z1 * dimsvolume.y + y0) * (dimsvolume.x / 2 + 1) + x1), c101 * val.x);
		atomicAdd(((tfloat*)(d_volumeft + (z1 * dimsvolume.y + y0) * (dimsvolume.x / 2 + 1) + x1)) + 1, c101 * val.y);
		atomicAdd((tfloat*)(d_samples + (z1 * dimsvolume.y + y0) * (dimsvolume.x / 2 + 1) + x1), c101 * weight);

		atomicAdd((tfloat*)(d_volumeft + (z1 * dimsvolume.y + y1) * (dimsvolume.x / 2 + 1) + x0), c011 * val.x);
		atomicAdd(((tfloat*)(d_volumeft + (z1 * dimsvolume.y + y1) * (dimsvolume.x / 2 + 1) + x0)) + 1, c011 * val.y);
		atomicAdd((tfloat*)(d_samples + (z1 * dimsvolume.y + y1) * (dimsvolume.x / 2 + 1) + x0), c011 * weight);

		atomicAdd((tfloat*)(d_volumeft + (z1 * dimsvolume.y + y1) * (dimsvolume.x / 2 + 1) + x1), c111 * val.x);
		atomicAdd(((tfloat*)(d_volumeft + (z1 * dimsvolume.y + y1) * (dimsvolume.x / 2 + 1) + x1)) + 1, c111 * val.y);
		atomicAdd((tfloat*)(d_samples + (z1 * dimsvolume.y + y1) * (dimsvolume.x / 2 + 1) + x1), c111 * weight);

		/*atomicAdd((tfloat*)(d_volumeft + (z0 * dimsvolume.y + y0) * (dimsvolume.x / 2 + 1) + x0), val.x);
		atomicAdd(((tfloat*)(d_volumeft + (z0 * dimsvolume.y + y0) * (dimsvolume.x / 2 + 1) + x0)) + 1, val.y);
		atomicAdd((tfloat*)(d_samples + (z0 * dimsvolume.y + y0) * (dimsvolume.x / 2 + 1) + x0), 1.0f);*/
	}
}