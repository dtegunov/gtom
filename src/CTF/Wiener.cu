#include "Prerequisites.cuh"
#include "CTF.cuh"
#include "FFT.cuh"
#include "Generics.cuh"
#include "Helper.cuh"
#include "Optimization.cuh"
#include "Transformation.cuh"


template<int ndims> __global__ void WienerPerFreqKernel(tcomplex* d_input, tfloat* d_fsc, tcomplex* d_output, tfloat* d_outputweights, int3 dims, CTFParamsLean p);


////////////////////////////////////////////////////////////
//Correct the CTF function to make all amplitudes positive//
////////////////////////////////////////////////////////////

void d_WienerPerFreq(tcomplex* d_input, int3 dimsinput, tfloat* d_fsc, CTFParams params, tcomplex* d_output, tfloat* d_outputweights)
{
	dim3 TpB = dim3(32, 8);
	dim3 grid = dim3((dimsinput.x / 2 + 1 + TpB.x - 1) / TpB.x, (dimsinput.y + TpB.y - 1) / TpB.y, dimsinput.z);
	if (DimensionCount(dimsinput) == 1)
		WienerPerFreqKernel<1> << <grid, TpB >> > (d_input, d_fsc, d_output, d_outputweights, dimsinput, CTFParamsLean(params));
	if (DimensionCount(dimsinput) == 2)
		WienerPerFreqKernel<2> << <grid, TpB >> > (d_input, d_fsc, d_output, d_outputweights, dimsinput, CTFParamsLean(params));
	else if (DimensionCount(dimsinput) == 3)
		WienerPerFreqKernel<3> << <grid, TpB >> > (d_input, d_fsc, d_output, d_outputweights, dimsinput, CTFParamsLean(params));
}


////////////////
//CUDA kernels//
////////////////

template<int ndims> __global__ void WienerPerFreqKernel(tcomplex* d_input, tfloat* d_fsc, tcomplex* d_output, tfloat* d_outputweights, int3 dims, CTFParamsLean p)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx > dims.x / 2)
		return;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	if (idy >= dims.y)
		return;
	int idz = blockIdx.z;

	double k, angle, radius;
	if (ndims == 1)
	{
		int x = -idx;

		angle = 0.0;
		radius = abs(x);
		k = radius * p.ny * 2.0 / (double)dims.x;
	}
	else if (ndims == 2)
	{
		int x = -idx;
		int y = dims.y / 2 - 1 - ((idy + dims.y / 2 - 1) % dims.y);

		float2 position = make_float2(x, y);
		angle = atan2(position.y, position.x);
		radius = sqrt(position.x * position.x + position.y * position.y);
		k = radius * p.ny * 2.0 / (double)dims.x;
	}
	else if (ndims == 3)
	{
		int x = -idx;
		int y = dims.y / 2 - 1 - ((idy + dims.y / 2 - 1) % dims.y);
		int z = dims.z / 2 - 1 - ((idz + dims.z / 2 - 1) % dims.z);

		float3 position = make_float3(x, y, z);
		radius = sqrt(position.x * position.x + position.y * position.y + position.z * position.z);
		k = radius * p.ny * 2.0 / (double)dims.x;
		angle = 0.0;
	}

	double amplitude = 0;

	if (radius <= dims.x / 2)
		amplitude = d_GetCTF<false>(k, angle, p);

	if (ndims == 1)
	{
		d_input += idx;
		d_output += idx;
		if (d_outputweights != NULL)
			d_outputweights += idx;
	}
	else if (ndims == 2)
	{
		d_input += idy * (dims.x / 2 + 1) + idx;
		d_output += idy * (dims.x / 2 + 1) + idx;
		if (d_outputweights != NULL)
			d_outputweights += idy * (dims.x / 2 + 1) + idx;
	}
	else if (ndims == 3)
	{
		d_input += (idz * dims.y + idy) * (dims.x / 2 + 1) + idx;
		d_output += (idz * dims.y + idy) * (dims.x / 2 + 1) + idx;
		if (d_outputweights != NULL)
			d_outputweights += (idz * dims.y + idy) * (dims.x / 2 + 1) + idx;
	}

	tcomplex input = *d_input;
	tfloat weight = (tfloat)0;
	int radiuslow = min(dims.x / 2 - 1, (int)radius);
	int radiushigh = min(dims.x / 2 - 1, radiuslow + 1);
	tfloat fsc = abs(lerp(d_fsc[radiuslow], d_fsc[radiushigh], radius - (tfloat)radiuslow));

	if (abs(amplitude) < 0.0001 || fsc < 0.001)
	{
		*d_output = make_cuComplex(0, 0);
		if (d_outputweights != NULL)
			*d_outputweights = 0;
	}
	else
	{
		weight = amplitude / (amplitude * amplitude + (1.0f - fsc) / fsc);
		*d_output = make_cuComplex(input.x * weight, input.y * weight);
		if (d_outputweights != NULL)
			*d_outputweights = amplitude * weight;
	}
}