#include "Prerequisites.cuh"
#include "CTF.cuh"
#include "FFT.cuh"
#include "Generics.cuh"
#include "Helper.cuh"
#include "Optimization.cuh"
#include "Transformation.cuh"


namespace gtom
{
	template<bool amplitudesquared> __global__ void CTFSimulateKernel(tfloat* d_output, float2* d_addresses, int n, CTFParamsLean* d_p);


	/////////////////////////////////////////////
	//Simulate the CTF function at given points//
	/////////////////////////////////////////////

	void d_CTFSimulate(CTFParams* h_params, float2* d_addresses, tfloat* d_output, uint n, bool amplitudesquared, int batch)
	{
		CTFParamsLean* h_lean;
		cudaMallocHost((void**)&h_lean, batch * sizeof(CTFParamsLean));
		for (int i = 0; i < batch; i++)
			h_lean[i] = CTFParamsLean(h_params[i]);
		CTFParamsLean* d_lean = (CTFParamsLean*)CudaMallocFromHostArray(h_lean, batch * sizeof(CTFParamsLean));
		cudaFreeHost(h_lean);

		int TpB = min(256, NextMultipleOf(n, 32));
		dim3 grid = dim3(NextMultipleOf(n, TpB), batch);
		if (amplitudesquared)
			CTFSimulateKernel<true> << <grid, TpB >> > (d_output, d_addresses, n, d_lean);
		else
			CTFSimulateKernel<false> << <grid, TpB >> > (d_output, d_addresses, n, d_lean);

		cudaFree(d_lean);
	}


	////////////////
	//CUDA kernels//
	////////////////

	template<bool amplitudesquared> __global__ void CTFSimulateKernel(tfloat* d_output, float2* d_addresses, int n, CTFParamsLean* d_p)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= n)
			return;

		CTFParamsLean p = d_p[blockIdx.y];
		float2 address = d_addresses[idx];
		float angle = address.y;
		double k = address.x * p.ny;

		d_output[blockIdx.y * n + idx] = d_GetCTF<amplitudesquared>(k, angle, p);
	}
}