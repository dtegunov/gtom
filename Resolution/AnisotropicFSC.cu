#include "../Prerequisites.cuh"
#include "../Functions.cuh"
#include "../GLMFunctions.cuh"


///////////////////////////
//CUDA kernel declaration//
///////////////////////////

template<uint maxshells, uint maxthreads, uint subdivs> __global__ void AnisotropicFSCKernel(tcomplex* d_volume1, tcomplex* d_volume2, int3 dimsvolume, int maxradius, int offset, tfloat3 direction, tfloat coneangle, tfloat falloff, uint elements, uint blockspervolume, tfloat* d_nums, tfloat* d_denoms1, tfloat* d_denoms2);


/////////////////////////////
//Fourier Shell Correlation//
/////////////////////////////

void d_AnisotropicFSC(tfloat* d_volume1, tfloat* d_volume2, int3 dimsvolume, tfloat* d_curve, int maxradius, tfloat3 direction, tfloat coneangle, tfloat falloff, cufftHandle* plan, int batch)
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
		AnisotropicFSCKernel<16, 256, 8> <<<grid, 256>>> (d_volumeft1, d_volumeft2, dimsvolume, maxradius, offset, direction, coneangle, falloff, threadspervolume, blockspervolume, d_nums, d_denoms1, d_denoms2);
	else if(maxradius <= 32)
		AnisotropicFSCKernel<32, 256, 16> <<<grid, 256>>> (d_volumeft1, d_volumeft2, dimsvolume, maxradius, offset, direction, coneangle, falloff, threadspervolume, blockspervolume, d_nums, d_denoms1, d_denoms2);
	else if(maxradius <= 64)
		AnisotropicFSCKernel<64, 256, 32> <<<grid, 256>>> (d_volumeft1, d_volumeft2, dimsvolume, maxradius, offset, direction, coneangle, falloff, threadspervolume, blockspervolume, d_nums, d_denoms1, d_denoms2);
	else if(maxradius <= 128)
		AnisotropicFSCKernel<128, 256, 64> <<<grid, 256>>> (d_volumeft1, d_volumeft2, dimsvolume, maxradius, offset, direction, coneangle, falloff, threadspervolume, blockspervolume, d_nums, d_denoms1, d_denoms2);
	else if(maxradius <= 256)
		AnisotropicFSCKernel<256, 256, 128> <<<grid, 256>>> (d_volumeft1, d_volumeft2, dimsvolume, maxradius, offset, direction, coneangle, falloff, threadspervolume, blockspervolume, d_nums, d_denoms1, d_denoms2);
	else if(maxradius <= 512)
		AnisotropicFSCKernel<512, 256, 256> <<<grid, 256>>> (d_volumeft1, d_volumeft2, dimsvolume, maxradius, offset, direction, coneangle, falloff, threadspervolume, blockspervolume, d_nums, d_denoms1, d_denoms2);
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

void d_AnisotropicFSCMap(tfloat* d_volume1, tfloat* d_volume2, int3 dimsvolume, tfloat* d_map, int2 anglesteps, int maxradius, tfloat threshold, cufftHandle* plan, int batch)
{
	float phistep = ToRad(360.0f) / (float)(anglesteps.x - 1);
	float thetastep = ToRad(90.0f) / (float)(anglesteps.y - 1);

	tfloat* d_curve = CudaMallocValueFilled(maxradius * sizeof(tfloat), (tfloat)0);

	for (int idtheta = 0; idtheta < anglesteps.y; idtheta++)
	{
		float theta = (float)idtheta * thetastep;
		float x = -cos(theta);

		for (int idphi = 0; idphi < anglesteps.x; idphi++)
		{
			float phi = (float)idphi * phistep;
			float z = cos(phi) * sin(theta);
			float y = sin(phi) * sin(theta);

			d_AnisotropicFSC(d_volume1, d_volume2, dimsvolume, d_curve, maxradius, tfloat3(x, y, z), min(phistep, thetastep), min(phistep, thetastep) * (tfloat)0.5, plan, batch);
			d_FirstIndexOf(d_curve, d_map + idtheta * anglesteps.x + idphi, maxradius, threshold, T_INTERP_LINEAR, batch);
		}
	}

	cudaFree(d_curve);
}


////////////////
//CUDA kernels//
////////////////

template<uint maxshells, uint maxthreads, uint subdivs> __global__ void AnisotropicFSCKernel(tcomplex* d_volume1, tcomplex* d_volume2, int3 dimsvolume, int maxradius, int offset, tfloat3 direction, tfloat coneangle, tfloat falloff, uint elements, uint blockspervolume, tfloat* d_nums, tfloat* d_denoms1, tfloat* d_denoms2)
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
	int y = idy - dimsvolume.y / 2;
	int z;

	if(id < elements)
	{
		if(x * x + y * y < maxradius * maxradius)
		{
			int zend = (maxradius) * ((threadIdx.x % subdivs) + 1) / subdivs;

			for (int idz = (maxradius) * (threadIdx.x % subdivs) / subdivs; idz < zend; idz++)
			{
				size_t address = ((idz + offset) * dimsvolume.y + idy) * (dimsvolume.x / 2 + 1) + idx;

				z = idz - dimsvolume.z / 2 + offset;

				tfloat radius = sqrt((tfloat)(x * x + y * y + z * z));
				if(radius >= maxradius)
					continue;

				glm::vec3 normdirection((tfloat)x / radius, (tfloat)y / radius, (tfloat)z / radius);
				tfloat angle = acos(abs(direction.x * normdirection.x + direction.y * normdirection.y + direction.z * normdirection.z));
				if(angle > coneangle + falloff)
					continue;

				tfloat angleweight = (tfloat)1;
				if(angle > coneangle)
					angleweight = (cos((angle - coneangle) / falloff * (tfloat)PI) + (tfloat)1.0) * (tfloat)0.5;

				int radiuslow = (int)radius;
				int radiushigh = min(maxradius - 1, radiuslow + 1);
				tfloat frachigh = radius - (tfloat)radiuslow;
				tfloat fraclow = (tfloat)1 - frachigh;

				tcomplex val1 = d_volume1[address];
				val1.x *= angleweight;
				val1.y *= angleweight;
				tfloat denomsval = val1.x * val1.x + val1.y * val1.y;
				//denoms1[maxshells * threadIdx.x + radiuslow] += denomsval;
				atomicAdd(denoms1 + maxshells * (threadIdx.x / subdivs) + radiuslow, denomsval * fraclow);
				atomicAdd(denoms1 + maxshells * (threadIdx.x / subdivs) + radiushigh, denomsval * frachigh);
				//denoms1[maxshells * (threadIdx.x / subdivs) + radiuslow] += (tfloat)1;
			
				tcomplex val2 = d_volume2[address];
				val2.x *= angleweight;
				val2.y *= angleweight;
				denomsval = val2.x * val2.x + val2.y * val2.y;
				//denoms2[maxshells * threadIdx.x + radiuslow] += denomsval;		
				atomicAdd(denoms2 + maxshells * (threadIdx.x / subdivs) + radiuslow, denomsval * fraclow);
				atomicAdd(denoms2 + maxshells * (threadIdx.x / subdivs) + radiushigh, denomsval * frachigh);
			
				denomsval = val1.x * val2.x + val1.y * val2.y;
				//nums[maxshells * threadIdx.x + radiuslow] += denomsval;
				//nums[maxshells * (threadIdx.x / subdivs) + radiuslow] += denomsval * fraclow;
				//nums[maxshells * (threadIdx.x / subdivs) + radiushigh] += denomsval * frachigh;
				atomicAdd(nums + maxshells * (threadIdx.x / subdivs) + radiuslow, denomsval * fraclow);
				atomicAdd(nums + maxshells * (threadIdx.x / subdivs) + radiushigh, denomsval * frachigh);
			}
		}
	}
	__syncthreads();

	if(id < elements)
	{
		if(x * x + y * y < maxradius * maxradius)
		{
			int zend = maxradius + (maxradius - 1) * ((threadIdx.x % subdivs) + 1) / subdivs;

			for (int idz = maxradius + (maxradius - 1) * (threadIdx.x % subdivs) / subdivs; idz < zend; idz++)
			{
				size_t address = ((idz + offset) * dimsvolume.y + idy) * (dimsvolume.x / 2 + 1) + idx;

				z = idz - dimsvolume.z / 2 + offset;

				tfloat radius = sqrt((tfloat)(x * x + y * y + z * z));
				if(radius >= maxradius)
					continue;

				glm::vec3 normdirection((tfloat)x / radius, (tfloat)y / radius, (tfloat)z / radius);
				tfloat angle = acos(abs(direction.x * normdirection.x + direction.y * normdirection.y + direction.z * normdirection.z));
				if(angle > coneangle + falloff)
					continue;

				tfloat angleweight = (tfloat)1;
				if(angle > coneangle)
					angleweight = (cos((angle - coneangle) / falloff * (tfloat)PI) + (tfloat)1.0) * (tfloat)0.5;

				int radiuslow = (int)radius;
				int radiushigh = min(maxradius - 1, radiuslow + 1);
				tfloat frachigh = radius - (tfloat)radiuslow;
				tfloat fraclow = (tfloat)1 - frachigh;

				tcomplex val1 = d_volume1[address];
				val1.x *= angleweight;
				val1.y *= angleweight;
				tfloat denomsval = val1.x * val1.x + val1.y * val1.y;
				//denoms1[maxshells * threadIdx.x + radiuslow] += denomsval;
				atomicAdd(denoms1 + maxshells * (threadIdx.x / subdivs) + radiuslow, denomsval * fraclow);
				atomicAdd(denoms1 + maxshells * (threadIdx.x / subdivs) + radiushigh, denomsval * frachigh);
				//denoms1[maxshells * (threadIdx.x / subdivs) + radiuslow] += (tfloat)1;
			
				tcomplex val2 = d_volume2[address];
				val2.x *= angleweight;
				val2.y *= angleweight;
				denomsval = val2.x * val2.x + val2.y * val2.y;
				//denoms2[maxshells * threadIdx.x + radiuslow] += denomsval;		
				atomicAdd(denoms2 + maxshells * (threadIdx.x / subdivs) + radiuslow, denomsval * fraclow);
				atomicAdd(denoms2 + maxshells * (threadIdx.x / subdivs) + radiushigh, denomsval * frachigh);
			
				denomsval = val1.x * val2.x + val1.y * val2.y;
				//nums[maxshells * threadIdx.x + radiuslow] += denomsval;
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