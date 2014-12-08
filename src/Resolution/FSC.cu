#include "Prerequisites.cuh"
#include "FFT.cuh"
#include "Generics.cuh"
#include "Helper.cuh"
#include "Resolution.cuh"


///////////////////////////
//CUDA kernel declaration//
///////////////////////////

__global__ void FSCKernel(tcomplex* d_volume1, tcomplex* d_volume2, int3 dimsvolume, int maxradius, tfloat* d_nums, tfloat* d_denoms1, tfloat* d_denoms2);


/////////////////////////////
//Fourier Shell Correlation//
/////////////////////////////

void d_FSC(tfloat* d_volume1, tfloat* d_volume2, int3 dimsvolume, tfloat* d_curve, int maxradius, cufftHandle* plan, int batch)
{
	cufftHandle localplanforw;
	if (plan == NULL)
		localplanforw = d_FFTR2CGetPlan(DimensionCount(dimsvolume), dimsvolume, batch);
	else
		localplanforw = *plan;

	tcomplex* d_volumeft1;
	cudaMalloc((void**)&d_volumeft1, ElementsFFT(dimsvolume) * batch * sizeof(tcomplex));
	d_FFTR2C(d_volume1, d_volumeft1, &localplanforw);

	tcomplex* d_volumeft2;
	cudaMalloc((void**)&d_volumeft2, ElementsFFT(dimsvolume) * batch * sizeof(tcomplex));
	d_FFTR2C(d_volume2, d_volumeft2, &localplanforw);

	d_FSC(d_volumeft1, d_volumeft2, dimsvolume, d_curve, maxradius, batch);

	cudaFree(d_volumeft1);
	cudaFree(d_volumeft2);
	if (plan == NULL)
		cufftDestroy(localplanforw);
}

void d_FSC(tcomplex* d_volumeft1, tcomplex* d_volumeft2, int3 dimsvolume, tfloat* d_curve, int maxradius, int batch)
{
	uint TpB = min(256, ElementsFFT(dimsvolume));
	dim3 grid = dim3(min((ElementsFFT(dimsvolume) + TpB - 1) / TpB, 128), batch);

	tfloat *d_nums, *d_denoms1, *d_denoms2;
	cudaMalloc((void**)&d_nums, maxradius * grid.x * batch * sizeof(tfloat));
	cudaMalloc((void**)&d_denoms1, maxradius * grid.x * batch * sizeof(tfloat));
	cudaMalloc((void**)&d_denoms2, maxradius * grid.x * batch * sizeof(tfloat));

	FSCKernel <<<grid, TpB>>> (d_volumeft1, d_volumeft2, dimsvolume, maxradius, d_nums, d_denoms1, d_denoms2);

	tfloat *d_rednums, *d_reddenoms1, *d_reddenoms2;
	cudaMalloc((void**)&d_rednums, maxradius * batch * sizeof(tfloat));
	cudaMalloc((void**)&d_reddenoms1, maxradius * batch * sizeof(tfloat));
	cudaMalloc((void**)&d_reddenoms2, maxradius * batch * sizeof(tfloat));

	d_ReduceAdd(d_nums, d_rednums, maxradius, grid.x, batch);
	d_ReduceAdd(d_denoms1, d_reddenoms1, maxradius, grid.x, batch);
	d_ReduceAdd(d_denoms2, d_reddenoms2, maxradius, grid.x, batch);

	cudaFree(d_denoms2);
	cudaFree(d_denoms1);
	cudaFree(d_nums);

	d_MultiplyByVector(d_reddenoms1, d_reddenoms2, d_reddenoms1, maxradius * batch);
	d_Sqrt(d_reddenoms1, d_reddenoms1, maxradius * batch);
	d_DivideSafeByVector(d_rednums, d_reddenoms1, d_curve, maxradius * batch);

	cudaFree(d_reddenoms2);
	cudaFree(d_reddenoms1);
	cudaFree(d_rednums);
}


////////////////
//CUDA kernels//
////////////////

__global__ void FSCKernel(tcomplex* d_volume1, tcomplex* d_volume2, int3 dimsvolume, int maxradius, tfloat* d_nums, tfloat* d_denoms1, tfloat* d_denoms2)
{
	__shared__ tfloat nums[512];
	__shared__ tfloat denoms1[512];
	__shared__ tfloat denoms2[512];

	d_volume1 += ElementsFFT(dimsvolume) * blockIdx.y;
	d_volume2 += ElementsFFT(dimsvolume) * blockIdx.y;

	for (int i = threadIdx.x; i < maxradius; i += blockDim.x)
	{
		nums[i] = 0;
		denoms1[i] = 0;
		denoms2[i] = 0;
	}
	__syncthreads();

	uint elementsrow = (dimsvolume.x / 2 + 1);
	uint elementsslice = elementsrow * dimsvolume.y;
	uint elementscube = elementsslice * dimsvolume.z;

	int maxradius2 = maxradius * maxradius;

	for (uint id = blockIdx.x * blockDim.x + threadIdx.x; id < elementscube; id += gridDim.x * blockDim.x)
	{
		int idz = id / elementsslice;
		int idy = (id - idz * elementsslice) / elementsrow;
		int idx = id % elementsrow;

		int rx = idx;
		int ry = dimsvolume.y / 2 - 1 - ((idy + dimsvolume.y / 2 - 1) % dimsvolume.y);
		int rz = dimsvolume.z / 2 - 1 - ((idz + dimsvolume.z / 2 - 1) % dimsvolume.z);
		int radius2 = rx * rx + ry * ry + rz * rz;
		if (radius2 >= maxradius2)
			continue;

		tfloat radius = sqrt((tfloat)radius2);

		int radiuslow = (int)radius;
		int radiushigh = min(maxradius - 1, radiuslow + 1);
		tfloat frachigh = radius - (tfloat)radiuslow;
		tfloat fraclow = (tfloat)1 - frachigh;

		tcomplex val1 = d_volume1[id];
		tfloat denomsval = val1.x * val1.x + val1.y * val1.y;
		atomicAdd(denoms1 + radiuslow, denomsval * fraclow);
		atomicAdd(denoms1 + radiushigh, denomsval * frachigh);

		tcomplex val2 = d_volume2[id];
		denomsval = val2.x * val2.x + val2.y * val2.y;
		atomicAdd(denoms2 + radiuslow, denomsval * fraclow);
		atomicAdd(denoms2 + radiushigh, denomsval * frachigh);

		denomsval = val1.x * val2.x + val1.y * val2.y;
		atomicAdd(nums + radiuslow, denomsval * fraclow);
		atomicAdd(nums + radiushigh, denomsval * frachigh);
	}
	__syncthreads();

	d_nums += maxradius * (blockIdx.y * gridDim.x + blockIdx.x);
	d_denoms1 += maxradius * (blockIdx.y * gridDim.x + blockIdx.x);
	d_denoms2 += maxradius * (blockIdx.y * gridDim.x + blockIdx.x);

	for (int i = threadIdx.x; i < maxradius; i += blockDim.x)
	{
		d_nums[i] = nums[i];
		d_denoms1[i] = denoms1[i];
		d_denoms2[i] = denoms2[i];
	}
}