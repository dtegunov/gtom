#include "Prerequisites.cuh"
#include "CTF.cuh"
#include "FFT.cuh"
#include "Generics.cuh"
#include "Helper.cuh"
#include "Optimization.cuh"
#include "Transformation.cuh"


template<int ndims> __global__ void CTFCorrectKernel(tcomplex* d_input, tcomplex* d_output, int3 dims, CTFParamsLean p);


////////////////////////////////////////////////////////////
//Correct the CTF function to make all amplitudes positive//
////////////////////////////////////////////////////////////

void d_CTFCorrect(tcomplex* d_input, int3 dimsinput, CTFParams params, tcomplex* d_output)
{
	dim3 TpB = dim3(32, 8);
	dim3 grid = dim3((dimsinput.x / 2 + 1 + TpB.x - 1) / TpB.x, (dimsinput.y + TpB.y - 1) / TpB.y, dimsinput.z);
	if (DimensionCount(dimsinput) == 2)
		CTFCorrectKernel<2> << <grid, TpB >> > (d_output, d_output, dimsinput, CTFParamsLean(params));
	else if (DimensionCount(dimsinput) == 3)
		CTFCorrectKernel<3> << <grid, TpB >> > (d_output, d_output, dimsinput, CTFParamsLean(params));
}


////////////////
//CUDA kernels//
////////////////

template<int ndims> __global__ void CTFCorrectKernel(tcomplex* d_input, tcomplex* d_output, int3 dims, CTFParamsLean p)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx > dims.x / 2)
		return;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	if (idy >= dims.y)
		return;
	int idz = blockIdx.z;

	double k, angle;
	if (ndims == 2)
	{
		int x = dims.x / 2 - idx;
		int y = dims.y - 1 - ((idy + dims.y / 2 - 1) % dims.y);

		float2 position = make_float2(x - dims.x / 2, y - dims.y / 2);
		angle = atan2(position.y, position.x);
		position = make_float2(position.x, position.y);
		k = sqrt(position.x * position.x + position.y * position.y) * p.ny * 2.0 / (double)dims.x;
	}
	else if (ndims == 3)
	{
		int x = dims.x / 2 - idx;
		int y = dims.y - 1 - ((idy + dims.y / 2 - 1) % dims.y);
		int z = dims.z - 1 - ((idz + dims.z / 2 - 1) % dims.z);

		float3 position = make_float3((float)(x - dims.x / 2), 
									  (float)(y - dims.y / 2), 
									  (float)(z - dims.z / 2));
		k = sqrt(position.x * position.x + position.y * position.y + position.z * position.z) * p.ny * 2.0 / (double)dims.x;
		angle = 0.0;
	}

	double amplitude = d_GetCTF<false>(k, angle, p);

	if (ndims == 2)
	{
		d_input += idy * (dims.x / 2 + 1) + idx;
		d_output += idy * (dims.x / 2 + 1) + idx;
	}
	else if (ndims == 3)
	{
		d_input += (idz * dims.y + idy) * (dims.x / 2 + 1) + idx;
		d_output += (idz * dims.y + idy) * (dims.x / 2 + 1) + idx;
	}

	tcomplex input = *d_input;
	if (amplitude < 0.0)
		*d_output = make_cuComplex(-input.x, -input.y);
	else
		*d_output = input;
}