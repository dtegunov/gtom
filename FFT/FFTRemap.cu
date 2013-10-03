#include "..\Prerequisites.cuh"
#include "..\Functions.cuh"

////////////////////////////
//CUDA kernel declarations//
////////////////////////////

__global__ void RemapFullToHalfFFTKernel(tfloat* d_input, tfloat* d_output, int3 dims);


////////////////
//Host methods//
////////////////

void d_RemapFullToHalfFFT(tfloat* d_input, tfloat* d_output, int3 dims)
{
	int TpB = 256;
	dim3 grid = dim3(((dims.x / 2 + 1) + TpB - 1) / TpB, dims.y, dims.z);
	RemapFullToHalfFFTKernel <<<grid, TpB>>> (d_input, d_output, dims);
}


////////////////
//CUDA kernels//
////////////////

__global__ void RemapFullToHalfFFTKernel(tfloat* d_input, tfloat* d_output, int3 dims)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if(x >= dims.x / 2 + 1)
		return;

	int rx = (x + (dims.x / 2)) % dims.x;
	int ry = ((blockIdx.y + ((dims.y + 1) / 2)) % dims.y);
	int rz = ((blockIdx.z + ((dims.z + 1) / 2)) % dims.z);

	d_output[(rz * dims.y + ry) * (dims.x / 2 + 1) + x] = d_input[(blockIdx.z * dims.y + blockIdx.y) * dims.x + rx];
}