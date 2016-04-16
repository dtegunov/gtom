#include "Prerequisites.cuh"
#include "CTF.cuh"
#include "Generics.cuh"
#include "Helper.cuh"
#include "ImageManipulation.cuh"
#include "Masking.cuh"


namespace gtom
{
	__global__ void DoseFilterKernel(tfloat* d_freq, tfloat* d_output, tfloat* d_dose, tfloat3 nikoconst, uint length);

	/////////////////////////////////////////////
	//Multiplies input by dose-dependent weight//
	/////////////////////////////////////////////

	void d_DoseFilter(tfloat* d_freq, tfloat* d_output, uint length, tfloat* h_dose, tfloat3 nikoconst, uint batch)
	{
		tfloat* h_doseconstant;
		cudaMallocHost((void**)&h_doseconstant, length * sizeof(tfloat));

		tfloat* d_dose = (tfloat*)CudaMallocFromHostArray(h_dose, batch * sizeof(tfloat));

		int TpB = tmin(128, NextMultipleOf(length, 32));
		dim3 grid = dim3((length + TpB - 1) / TpB, batch, 1);
		DoseFilterKernel << <grid, TpB >> > (d_freq, d_output, d_dose, nikoconst, length);

		cudaFree(d_dose);
	}

	__global__ void DoseFilterKernel(tfloat* d_freq, tfloat* d_output, tfloat* d_dose, tfloat3 nikoconst, uint length)
	{
		d_output += blockIdx.y * length;
		tfloat dose = d_dose[blockIdx.y];

		for (uint i = blockIdx.x * blockDim.x + threadIdx.x; 
			 i < length; 
			 i += gridDim.x * blockDim.x)
			 d_output[i] = exp(dose * (-(tfloat)1 / ((tfloat)2 * nikoconst.x * pow(d_freq[i], nikoconst.y) + nikoconst.z)));
	}
}