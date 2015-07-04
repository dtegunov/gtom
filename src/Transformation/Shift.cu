#include "Prerequisites.cuh"
#include "FFT.cuh"
#include "Helper.cuh"
#include "Transformation.cuh"


namespace gtom
{
	////////////////////////////
	//CUDA kernel declarations//
	////////////////////////////

	template<int ndims, bool iszerocentered> __global__ void ShiftFourierKernel(tcomplex* d_input, tcomplex* d_output, int3 dims, tfloat3* d_delta);


	////////////////////////////////////////
	//Equivalent of TOM's tom_shift method//
	////////////////////////////////////////

	void d_Shift(tfloat* d_input, tfloat* d_output, int3 dims, tfloat3* h_delta, cufftHandle* planforw, cufftHandle* planback, tcomplex* d_sharedintermediate, int batch)
	{
		tcomplex* d_intermediate = NULL;
		if (d_sharedintermediate == NULL)
			cudaMalloc((void**)&d_intermediate, batch * ElementsFFT(dims) * sizeof(tcomplex));
		else
			d_intermediate = d_sharedintermediate;

		if (planforw == NULL)
			d_FFTR2C(d_input, d_intermediate, DimensionCount(dims), dims, batch);
		else
			d_FFTR2C(d_input, d_intermediate, planforw);

		d_Shift(d_intermediate, d_intermediate, dims, h_delta, false, batch);

		if (planback == NULL)
			d_IFFTC2R(d_intermediate, d_output, DimensionCount(dims), dims, batch);
		else
			d_IFFTC2R(d_intermediate, d_output, planback, dims);

		if (d_sharedintermediate == NULL)
			cudaFree(d_intermediate);
	}

	void d_Shift(tcomplex* d_input, tcomplex* d_output, int3 dims, tfloat3* h_delta, bool iszerocentered, int batch)
	{
		tfloat3* h_deltanorm = (tfloat3*)malloc(batch * sizeof(tfloat3));
		for (int b = 0; b < batch; b++)
			h_deltanorm[b] = tfloat3(h_delta[b].x / (tfloat)dims.x, h_delta[b].y / (tfloat)dims.y, h_delta[b].z / (tfloat)dims.z);
		tfloat3* d_delta = (tfloat3*)CudaMallocFromHostArray(h_deltanorm, batch * sizeof(tfloat3));
		free(h_deltanorm);

		int TpB = tmin(256, NextMultipleOf(dims.x / 2 + 1, 32));
		dim3 grid = dim3(dims.y, dims.z, batch);
		if (!iszerocentered)
		{
			if (DimensionCount(dims) == 3)
				ShiftFourierKernel <3, false> << <grid, TpB >> > (d_input, d_output, dims, d_delta);
			else
				ShiftFourierKernel <2, false> << <grid, TpB >> > (d_input, d_output, dims, d_delta);
		}
		else
		{
			if (DimensionCount(dims) == 3)
				ShiftFourierKernel <3, true> << <grid, TpB >> > (d_input, d_output, dims, d_delta);
			else
				ShiftFourierKernel <2, true> << <grid, TpB >> > (d_input, d_output, dims, d_delta);
		}

		cudaFree(d_delta);
	}


	////////////////
	//CUDA kernels//
	////////////////

	template<int ndims, bool iszerocentered> __global__ void ShiftFourierKernel(tcomplex* d_input, tcomplex* d_output, int3 dims, tfloat3* d_delta)
	{
		int idy = blockIdx.x;
		int idz = blockIdx.y;

		int x, y, z;
		if (!iszerocentered)
		{
			y = FFTShift(idy, dims.y) - dims.y / 2;
			z = FFTShift(idz, dims.z) - dims.z / 2;
		}
		else
		{
			y = dims.y / 2 - idy;
			z = dims.z / 2 - idz;
		}

		d_input += ((blockIdx.z * dims.z + idz) * dims.y + idy) * (dims.x / 2 + 1);
		d_output += ((blockIdx.z * dims.z + idz) * dims.y + idy) * (dims.x / 2 + 1);
		tfloat3 delta = d_delta[blockIdx.z];

		for (int idx = threadIdx.x; idx <= dims.x / 2; idx += blockDim.x)
		{
			if (!iszerocentered)
				x = FFTShift(idx, dims.x) - dims.x / 2;
			else
				x = dims.x / 2 - idx;

			tfloat factor = (delta.x * (tfloat)x + delta.y * (tfloat)y + (ndims > 2 ? delta.z * (tfloat)z : (tfloat)0)) * (tfloat)PI2;
			tcomplex multiplicator = make_cuComplex(cos(factor), sin(-factor));

			d_output[idx] = cmul(d_input[idx], multiplicator);
		}
	}
}