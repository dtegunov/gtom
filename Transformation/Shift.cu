#include "..\Prerequisites.cuh"
#include "..\Functions.cuh"


////////////////////////////
//CUDA kernel declarations//
////////////////////////////

__global__ void ShiftFourierKernel(tcomplex* d_input, tcomplex* d_output, int3 dims, tfloat3 delta);
__global__ void ShiftIntegerKernel(tfloat* d_input, tfloat* d_output, int3 dims, int3 delta);


/////////////////////////////////////////////
//Equivalent of TOM's tom_cart2polar method//
/////////////////////////////////////////////

void d_Shift(tfloat* d_input, tfloat* d_output, int3 dims, tfloat3* delta, int batch)
{
	size_t elements = dims.x * dims.y * dims.z;
	size_t elementsFFT = (dims.x / 2 + 1) * dims.y * dims.z;

	tcomplex* d_intermediate;
	cudaMalloc((void**)&d_intermediate, elementsFFT * sizeof(tcomplex));

	for (int b = 0; b < batch; b++)
	{
		if(fmod(delta[b].x, (tfloat)1) != (tfloat)0 || fmod(delta[b].y, (tfloat)1) != (tfloat)0 || fmod(delta[b].z, (tfloat)1) != (tfloat)0)
		{
			tfloat3 normdelta = tfloat3(delta[b].x / (tfloat)dims.x, delta[b].y / (tfloat)dims.y, delta[b].z / (tfloat)dims.z);

			d_FFTR2C(d_input + elements * b, d_intermediate, DimensionCount(dims), dims);

			int TpB = min(256, NextMultipleOf(dims.x / 2 + 1, 32));
			dim3 grid = dim3(((dims.x / 2 + 1) + TpB - 1) / TpB, dims.y, dims.z);
			ShiftFourierKernel <<<grid, TpB>>> (d_intermediate, d_intermediate, dims, normdelta);
			
			d_IFFTC2R(d_intermediate, d_output + elements * b, DimensionCount(dims), dims);
		}
		else
		{
			int TpB = min(256, NextMultipleOf(dims.x, 32));
			dim3 grid = dim3((dims.x + TpB - 1) / TpB, dims.y, dims.z);
			ShiftIntegerKernel <<<grid, TpB>>> (d_input + elements * b, d_input == d_output ? (tfloat*)d_intermediate : (d_output + elements * b), dims, toInt3((int)delta[b].x, (int)delta[b].y, (int)delta[b].z));

			if(d_input == d_output)
				cudaMemcpy(d_output + elements * b, d_intermediate, elements * sizeof(tfloat), cudaMemcpyDeviceToDevice);
		}
	}

	cudaFree(d_intermediate);
}


////////////////
//CUDA kernels//
////////////////

__global__ void ShiftFourierKernel(tcomplex* d_input, tcomplex* d_output, int3 dims, tfloat3 delta)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if(x >= dims.x / 2 + 1)
		return;
	if(x == dims.x / 2)
		x = (-x);
	int y = ((blockIdx.y + ((dims.y + 1) / 2)) % dims.y) - (dims.y / 2);
	int z = ((blockIdx.z + ((dims.z + 1) / 2)) % dims.z) - (dims.z / 2);

	tfloat factorx = delta.x * (tfloat)x * (tfloat)PI2;
	tcomplex multx = { cos(factorx), sin(-factorx) };
	if(blockIdx.x * blockDim.x + threadIdx.x == dims.x / 2)
		multx.y = (tfloat)0;
	tfloat factory = delta.y * (tfloat)y * (tfloat)PI2;
	tcomplex multy = { cos(factory), sin(-factory) };
	if(blockIdx.y == dims.y / 2)
		multy.y = (tfloat)0;

	tcomplex multiplicator = cmul(multx, multy);

	if(dims.z > 1)
	{
		tfloat factorz = delta.z * (tfloat)z * (tfloat)PI2;
		tcomplex multz = { cos(factorz), sin(-factorz) };
		if(blockIdx.z == dims.z / 2)
			multz.y = (tfloat)0;
		multiplicator = cmul(multiplicator, multz);
	}	

	size_t id = (blockIdx.z * dims.y + blockIdx.y) * (dims.x / 2 + 1) + blockIdx.x * blockDim.x + threadIdx.x;
	d_output[id] = cmul(d_input[id], multiplicator);
	//d_output[id] = multiplicator;
}

__global__ void ShiftIntegerKernel(tfloat* d_input, tfloat* d_output, int3 dims, int3 delta)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	d_output[((abs(dims.z + (int)blockIdx.z) % dims.z) * dims.y + (abs(dims.y + (int)blockIdx.y) % dims.y)) * dims.x + (abs(dims.x + x) % dims.x)] = d_input[(blockIdx.z * dims.y + blockIdx.y) * dims.x + x];
}