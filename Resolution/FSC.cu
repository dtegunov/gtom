#include "../Prerequisites.cuh"
#include "../Functions.cuh"


///////////////////////////
//CUDA kernel declaration//
///////////////////////////

template<uint maxshells, uint maxthreads, uint subdivs> __global__ void FSCKernel(tcomplex* d_volume1, tcomplex* d_volume2, int3 dimsvolume, int maxradius, int offset, uint elements, uint blockspervolume, tfloat* d_nums, tfloat* d_denoms1, tfloat* d_denoms2);


/////////////////////////////
//Fourier Shell Correlation//
/////////////////////////////

void d_FSC(tfloat* d_volume1, tfloat* d_volume2, int3 dimsvolume, tfloat* d_curve, int maxradius, cufftHandle* plan, int batch)
{
	if(dimsvolume.x != dimsvolume.y || dimsvolume.x != dimsvolume.z)
		throw;

	cufftHandle localplanforw;
	if(plan == NULL)
		localplanforw = d_FFTR2CGetPlan(DimensionCount(dimsvolume), dimsvolume, batch);
	else
		localplanforw = *plan;

	tcomplex* d_volumeft1;
	cudaMalloc((void**)&d_volumeft1, ElementsFFT(dimsvolume) * batch * sizeof(tcomplex));
	d_FFTR2C(d_volume1, d_volumeft1, &localplanforw);

	tcomplex* d_volumeft2;
	cudaMalloc((void**)&d_volumeft2, ElementsFFT(dimsvolume) * batch * sizeof(tcomplex));
	d_FFTR2C(d_volume2, d_volumeft2, &localplanforw);

	tcomplex* d_temp;
	cudaMalloc((void**)&d_temp, ElementsFFT(dimsvolume) * batch * sizeof(tcomplex));

	d_RemapHalfFFT2Half(d_volumeft1, d_temp, dimsvolume, batch);
	cudaMemcpy(d_volumeft1, d_temp, ElementsFFT(dimsvolume) * batch * sizeof(tcomplex), cudaMemcpyDeviceToDevice);
	d_RemapHalfFFT2Half(d_volumeft2, d_temp, dimsvolume, batch);
	cudaMemcpy(d_volumeft2, d_temp, ElementsFFT(dimsvolume) * batch * sizeof(tcomplex), cudaMemcpyDeviceToDevice);

	cudaFree(d_temp);

	int maxthreads;
	if(maxradius <= 16)
		maxthreads = 32 / 1;
	else if(maxradius <= 32)
		maxthreads = 256 / 16;
	else if(maxradius <= 64)
		maxthreads = 64 / 8;
	else if(maxradius <= 128)
		maxthreads = 64 / 16;
	else if(maxradius <= 256)
		maxthreads = 64 / 32;
	else if(maxradius <= 512)
		maxthreads = 32 / 8;

	int threadspervolume = maxradius * (maxradius * 2 - 1);
	int blockspervolume = (threadspervolume + maxthreads - 1) / maxthreads;

	int offset = dimsvolume.x / 2 - maxradius + 1;

	tfloat* d_nums = CudaMallocValueFilled(maxradius * blockspervolume * batch, (tfloat)0);
	tfloat* d_denoms1 = CudaMallocValueFilled(maxradius * blockspervolume * batch, (tfloat)0);
	tfloat* d_denoms2 = CudaMallocValueFilled(maxradius * blockspervolume * batch, (tfloat)0);

	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	dim3 grid = dim3(min(blockspervolume, 32786), (blockspervolume + 32767) / 32768, batch);
	if(maxradius <= 16)
		FSCKernel<16, 256, 8> <<<grid, 256>>> (d_volumeft1, d_volumeft2, dimsvolume, maxradius, offset, threadspervolume, blockspervolume, d_nums, d_denoms1, d_denoms2);
	else if(maxradius <= 32)
		FSCKernel<32, 256, 16> <<<grid, 256>>> (d_volumeft1, d_volumeft2, dimsvolume, maxradius, offset, threadspervolume, blockspervolume, d_nums, d_denoms1, d_denoms2);
	else if(maxradius <= 64)
		FSCKernel<64, 256, 32> <<<grid, 256>>> (d_volumeft1, d_volumeft2, dimsvolume, maxradius, offset, threadspervolume, blockspervolume, d_nums, d_denoms1, d_denoms2);
	else if(maxradius <= 128)
		FSCKernel<128, 256, 64> <<<grid, 256>>> (d_volumeft1, d_volumeft2, dimsvolume, maxradius, offset, threadspervolume, blockspervolume, d_nums, d_denoms1, d_denoms2);
	else if(maxradius <= 256)
		FSCKernel<256, 256, 128> <<<grid, 256>>> (d_volumeft1, d_volumeft2, dimsvolume, maxradius, offset, threadspervolume, blockspervolume, d_nums, d_denoms1, d_denoms2);
	else if(maxradius <= 512)
		FSCKernel<512, 256, 256> <<<grid, 256>>> (d_volumeft1, d_volumeft2, dimsvolume, maxradius, offset, threadspervolume, blockspervolume, d_nums, d_denoms1, d_denoms2);
	else
		throw;

	tfloat* h_nums = (tfloat*)MallocFromDeviceArray(d_nums, maxradius * blockspervolume * batch * sizeof(tfloat));
	free(h_nums);

	tfloat *d_rednums, *d_reddenoms1, *d_reddenoms2;
	cudaMalloc((void**)&d_rednums, maxradius * batch * sizeof(tfloat));
	cudaMalloc((void**)&d_reddenoms1, maxradius * batch * sizeof(tfloat));
	cudaMalloc((void**)&d_reddenoms2, maxradius * batch * sizeof(tfloat));

	d_ReduceAdd(d_nums, d_rednums, maxradius, blockspervolume, batch);
	d_ReduceAdd(d_denoms1, d_reddenoms1, maxradius, blockspervolume, batch);
	d_ReduceAdd(d_denoms2, d_reddenoms2, maxradius, blockspervolume, batch);

	//tfloat* h_reddenoms1 = (tfloat*)MallocFromDeviceArray(d_reddenoms1, maxradius * batch * sizeof(tfloat));
	//free(h_reddenoms1);

	cudaFree(d_denoms2);
	cudaFree(d_denoms1);
	cudaFree(d_nums);

	d_MultiplyByVector(d_reddenoms1, d_reddenoms2, d_reddenoms1, maxradius * batch);
	d_Sqrt(d_reddenoms1, d_reddenoms1, maxradius * batch);
	d_AddScalar(d_reddenoms1, d_reddenoms1, maxradius * batch, (tfloat)0.000001);
	d_DivideByVector(d_rednums, d_reddenoms1, d_curve, maxradius * batch);

	cudaFree(d_reddenoms2);
	cudaFree(d_reddenoms1);
	cudaFree(d_rednums);
	cudaFree(d_volumeft1);
	cudaFree(d_volumeft2);

	if(plan == NULL)
		cufftDestroy(localplanforw);
}


////////////////
//CUDA kernels//
////////////////

template<uint maxshells, uint maxthreads, uint subdivs> __global__ void FSCKernel(tcomplex* d_volume1, tcomplex* d_volume2, int3 dimsvolume, int maxradius, int offset, uint elements, uint blockspervolume, tfloat* d_nums, tfloat* d_denoms1, tfloat* d_denoms2)
{
	__shared__ tfloat nums[maxshells * maxthreads / subdivs];
	__shared__ tfloat denoms1[maxshells * maxthreads / subdivs];
	__shared__ tfloat denoms2[maxshells * maxthreads / subdivs];

	d_volume1 += ElementsFFT(dimsvolume) * blockIdx.z;
	d_volume2 += ElementsFFT(dimsvolume) * blockIdx.z;

	uint id = (blockIdx.y * gridDim.x + blockIdx.x) * (maxthreads / subdivs) + (threadIdx.x / subdivs);

	if(threadIdx.x % subdivs == 0)
		for (int i = 0; i < maxradius; i++)
		{
			nums[maxshells * (threadIdx.x / subdivs) + i] = 0;
			denoms1[maxshells * (threadIdx.x / subdivs) + i] = 0;
			denoms2[maxshells * (threadIdx.x / subdivs) + i] = 0;
		}
	__syncthreads();

	int idx = (id % (uint)maxradius) + offset;
	int idy = (id / (uint)maxradius) + offset;

	int x = idx - dimsvolume.x / 2;
	x *= x;
	int y = idy - dimsvolume.y / 2;
	y *= y;
	int z;

	if(id < elements)
	{
		if(x + y < maxradius * maxradius)
		{
			int zend = (maxradius) * ((threadIdx.x % subdivs) + 1) / subdivs;

			for (int idz = (maxradius) * (threadIdx.x % subdivs) / subdivs; idz < zend; idz++)
			{
				size_t address = ((idz + offset) * dimsvolume.y + idy) * (dimsvolume.x / 2 + 1) + idx;

				z = idz - dimsvolume.z / 2 + offset;
				z *= z;

				tfloat radius = sqrt((tfloat)(x + y + z));
				if(radius >= maxradius)
					continue;

				int radiuslow = (int)radius;
				int radiushigh = min(maxradius - 1, radiuslow + 1);
				tfloat frachigh = radius - (tfloat)radiuslow;
				tfloat fraclow = (tfloat)1 - frachigh;

				tcomplex val1 = d_volume1[address];
				tfloat denomsval = val1.x * val1.x + val1.y * val1.y;
				atomicAdd(denoms1 + maxshells * (threadIdx.x / subdivs) + radiuslow, denomsval * fraclow);
				atomicAdd(denoms1 + maxshells * (threadIdx.x / subdivs) + radiushigh, denomsval * frachigh);
				//denoms1[maxshells * (threadIdx.x / subdivs) + radiuslow] += (tfloat)1;
			
				tcomplex val2 = d_volume2[address];
				denomsval = val2.x * val2.x + val2.y * val2.y;
				atomicAdd(denoms2 + maxshells * (threadIdx.x / subdivs) + radiuslow, denomsval * fraclow);
				atomicAdd(denoms2 + maxshells * (threadIdx.x / subdivs) + radiushigh, denomsval * frachigh);
			
				denomsval = val1.x * val2.x + val1.y * val2.y;
				atomicAdd(nums + maxshells * (threadIdx.x / subdivs) + radiuslow, denomsval * fraclow);
				atomicAdd(nums + maxshells * (threadIdx.x / subdivs) + radiushigh, denomsval * frachigh);
			}
		}
	}
	__syncthreads();

	if(id < elements)
	{
		if(x + y < maxradius * maxradius)
		{
			int zend = maxradius + (maxradius - 1) * ((threadIdx.x % subdivs) + 1) / subdivs;

			for (int idz = maxradius + (maxradius - 1) * (threadIdx.x % subdivs) / subdivs; idz < zend; idz++)
			{
				size_t address = ((idz + offset) * dimsvolume.y + idy) * (dimsvolume.x / 2 + 1) + idx;

				z = idz - dimsvolume.z / 2 + offset;
				z *= z;

				tfloat radius = sqrt((tfloat)(x + y + z));
				if(radius >= maxradius)
					continue;

				int radiuslow = (int)radius;
				int radiushigh = min(maxradius - 1, radiuslow + 1);
				tfloat frachigh = radius - (tfloat)radiuslow;
				tfloat fraclow = (tfloat)1 - frachigh;

				tcomplex val1 = d_volume1[address];
				tfloat denomsval = val1.x * val1.x + val1.y * val1.y;
				atomicAdd(denoms1 + maxshells * (threadIdx.x / subdivs) + radiuslow, denomsval * fraclow);
				atomicAdd(denoms1 + maxshells * (threadIdx.x / subdivs) + radiushigh, denomsval * frachigh);
				//denoms1[maxshells * (threadIdx.x / subdivs) + radiuslow] += (tfloat)1;
			
				tcomplex val2 = d_volume2[address];
				denomsval = val2.x * val2.x + val2.y * val2.y;
				atomicAdd(denoms2 + maxshells * (threadIdx.x / subdivs) + radiuslow, denomsval * fraclow);
				atomicAdd(denoms2 + maxshells * (threadIdx.x / subdivs) + radiushigh, denomsval * frachigh);
			
				denomsval = val1.x * val2.x + val1.y * val2.y;
				atomicAdd(nums + maxshells * (threadIdx.x / subdivs) + radiuslow, denomsval * fraclow);
				atomicAdd(nums + maxshells * (threadIdx.x / subdivs) + radiushigh, denomsval * frachigh);
			}
		}
	}
	__syncthreads();

	for (int i = threadIdx.x; i < maxradius; i += blockDim.x)
	{
		for (int t = 1; t < maxthreads / subdivs; t++)
		{
			nums[i] += nums[t * maxshells + i];
			denoms1[i] += denoms1[t * maxshells + i];
			denoms2[i] += denoms2[t * maxshells + i];
		}
	}
	__syncthreads();

	if(blockIdx.y * gridDim.x + blockIdx.x >= blockspervolume)
		return;

	d_nums += maxradius * (blockspervolume * blockIdx.z + (blockIdx.y * gridDim.x + blockIdx.x));
	d_denoms1 += maxradius * (blockspervolume * blockIdx.z + (blockIdx.y * gridDim.x + blockIdx.x));
	d_denoms2 += maxradius * (blockspervolume * blockIdx.z + (blockIdx.y * gridDim.x + blockIdx.x));

	for (int i = threadIdx.x; i < maxradius; i += blockDim.x)
	{
		d_nums[i] += nums[i];
		d_denoms1[i] += denoms1[i];
		d_denoms2[i] += denoms2[i];
	}
}