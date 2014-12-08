#include "Prerequisites.cuh"
#include "CTF.cuh"
#include "FFT.cuh"
#include "Generics.cuh"
#include "Helper.cuh"
#include "Optimization.cuh"
#include "Transformation.cuh"


template<bool amplitudesquared> __global__ void CTFSimulateKernel(tfloat* d_output, float2* d_addresses, int n, CTFParamsLean p);


/////////////////////////////////////////////
//Simulate the CTF function at given points//
/////////////////////////////////////////////

void d_CTFSimulate(CTFParams params, float2* d_addresses, tfloat* d_output, uint n, bool amplitudesquared)
{
	int TpB = min(256, NextMultipleOf(n, 32));
	int grid = (n + TpB - 1) / TpB;
	if (amplitudesquared)
		CTFSimulateKernel<true> << <grid, TpB >> > (d_output, d_addresses, n, CTFParamsLean(params));
	else
		CTFSimulateKernel<false> << <grid, TpB >> > (d_output, d_addresses, n, CTFParamsLean(params));
}


////////////////
//CUDA kernels//
////////////////

template<bool amplitudesquared> __global__ void CTFSimulateKernel(tfloat* d_output, float2* d_addresses, int n, CTFParamsLean p)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= n)
		return;

	float2 address = d_addresses[idx];
	float angle = address.y;
	double k = address.x * p.ny;	

	d_output[idx] = d_GetCTF<amplitudesquared>(k, angle, p);
}