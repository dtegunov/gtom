#include "Prerequisites.cuh"
#include "CTF.cuh"
#include "FFT.cuh"
#include "Generics.cuh"
#include "Helper.cuh"
#include "Optimization.cuh"
#include "Transformation.cuh"


namespace gtom
{
	template<bool amplitudesquared, bool ignorefirstpeak> __global__ void CTFSimulateKernel(tfloat* d_output, float2* d_addresses, int n, CTFParamsLean* d_p);
	template<bool amplitudesquared, bool ignorefirstpeak> __global__ void CTFSimulateKernel(half* d_output, half2* d_addresses, int n, CTFParamsLean* d_p);


	/////////////////////////////////////////////
	//Simulate the CTF function at given points//
	/////////////////////////////////////////////

	void d_CTFSimulate(CTFParams* h_params, float2* d_addresses, tfloat* d_output, uint n, bool amplitudesquared, bool ignorefirstpeak, int batch)
	{
		CTFParamsLean* h_lean;
		cudaMallocHost((void**)&h_lean, batch * sizeof(CTFParamsLean));
		#pragma omp parallel for
		for (int i = 0; i < batch; i++)
			h_lean[i] = CTFParamsLean(h_params[i], toInt3(1, 1, 1));	// Sidelength and pixelsize are already included in d_addresses
		CTFParamsLean* d_lean = (CTFParamsLean*)CudaMallocFromHostArray(h_lean, batch * sizeof(CTFParamsLean));
		cudaFreeHost(h_lean);

		int TpB = tmin(128, NextMultipleOf(n, 32));
		dim3 grid = dim3(tmin(batch > 1 ? 16 : 128, (n + TpB - 1) / TpB), batch);
		if (amplitudesquared)
			if (ignorefirstpeak)
				CTFSimulateKernel<true, true> << <grid, TpB >> > (d_output, d_addresses, n, d_lean);
			else
				CTFSimulateKernel<true, false> << <grid, TpB >> > (d_output, d_addresses, n, d_lean);
		else
			if (ignorefirstpeak)
				CTFSimulateKernel<false, true> << <grid, TpB >> > (d_output, d_addresses, n, d_lean);
			else
				CTFSimulateKernel<false, false> << <grid, TpB >> > (d_output, d_addresses, n, d_lean);

		cudaFree(d_lean);
	}

	void d_CTFSimulate(CTFParams* h_params, half2* d_addresses, half* d_output, uint n, bool amplitudesquared, bool ignorefirstpeak, int batch)
	{
		CTFParamsLean* h_lean;
		cudaMallocHost((void**)&h_lean, batch * sizeof(CTFParamsLean));
		#pragma omp parallel for
		for (int i = 0; i < batch; i++)
			h_lean[i] = CTFParamsLean(h_params[i], toInt3(1, 1, 1));	// Sidelength and pixelsize are already included in d_addresses
		CTFParamsLean* d_lean = (CTFParamsLean*)CudaMallocFromHostArray(h_lean, batch * sizeof(CTFParamsLean));
		cudaFreeHost(h_lean);

		int TpB = tmin(128, NextMultipleOf(n, 32));
		dim3 grid = dim3(tmin(batch > 1 ? 16 : 128, (n + TpB - 1) / TpB), batch);
		if (amplitudesquared)
			if (ignorefirstpeak)
				CTFSimulateKernel<true, true> << <grid, TpB >> > (d_output, d_addresses, n, d_lean);
			else
				CTFSimulateKernel<true, false> << <grid, TpB >> > (d_output, d_addresses, n, d_lean);
		else
			if (ignorefirstpeak)
				CTFSimulateKernel<false, true> << <grid, TpB >> > (d_output, d_addresses, n, d_lean);
			else
				CTFSimulateKernel<false, false> << <grid, TpB >> > (d_output, d_addresses, n, d_lean);

		cudaFree(d_lean);
	}


	////////////////
	//CUDA kernels//
	////////////////

	template<bool amplitudesquared, bool ignorefirstpeak> __global__ void CTFSimulateKernel(tfloat* d_output, float2* d_addresses, int n, CTFParamsLean* d_p)
	{
		CTFParamsLean p = d_p[blockIdx.y];
		d_output += blockIdx.y * n;

		for (uint idx = blockIdx.x * blockDim.x + threadIdx.x;
			 idx < n;
			 idx += gridDim.x * blockDim.x)
		{
			float2 address = d_addresses[idx];
			float angle = address.y;
			float k = address.x;

			float pixelsize = p.pixelsize + p.pixeldelta * __cosf(2.0f * (angle - p.pixelangle));
			k /= pixelsize;

			d_output[idx] = d_GetCTF<amplitudesquared, ignorefirstpeak>(k, angle, p);
		}
	}

	template<bool amplitudesquared, bool ignorefirstpeak> __global__ void CTFSimulateKernel(half* d_output, half2* d_addresses, int n, CTFParamsLean* d_p)
	{
		CTFParamsLean p = d_p[blockIdx.y];
		d_output += blockIdx.y * n;

		for (uint idx = blockIdx.x * blockDim.x + threadIdx.x;
			idx < n;
			idx += gridDim.x * blockDim.x)
		{
			float2 address = __half22float2(d_addresses[idx]);
			float angle = address.y;
			float k = address.x;

			float pixelsize = p.pixelsize + p.pixeldelta * __cosf(2.0f * (angle - p.pixelangle));
			k /= pixelsize;

			d_output[idx] = __float2half(d_GetCTF<amplitudesquared, ignorefirstpeak>(k, angle, p));
		}
	}
}