#include "..\Prerequisites.cuh"
#include "..\Functions.cuh"

__global__ void MultiplyKernel(tfloat* d_input, tfloat* d_output, size_t elements, tfloat multiplicator);
__global__ void AddKernel(tfloat* d_input, tfloat* d_output, size_t elements, tfloat summand);

//////////////////
//Multiplication//
//////////////////

void d_Multiply(tfloat* d_input, tfloat* d_output, size_t elements, tfloat multiplicator)
{
	size_t TpB = 256;
	size_t totalblocks = min((elements + TpB - 1) / TpB, 8192);
	dim3 grid = dim3(totalblocks);
	MultiplyKernel <<<grid, TpB>>> (d_input, d_output, elements, multiplicator);
}

__global__ void MultiplyKernel(tfloat* d_input, tfloat* d_output, size_t elements, tfloat multiplicator)
{
	for(size_t id = blockIdx.x * blockDim.x + threadIdx.x; 
		id < elements; 
		id += blockDim.x * gridDim.x)
		d_output[id] = d_input[id] * multiplicator;
}


////////////
//Addition//
////////////

void d_Add(tfloat* d_input, tfloat* d_output, size_t elements, tfloat summand)
{
	size_t TpB = 256;
	size_t totalblocks = min((elements + TpB - 1) / TpB, 8192);
	dim3 grid = dim3(totalblocks);
	AddKernel <<<grid, TpB>>> (d_input, d_output, elements, summand);
}

__global__ void AddKernel(tfloat* d_input, tfloat* d_output, size_t elements, tfloat summand)
{
	for(size_t id = blockIdx.x * blockDim.x + threadIdx.x; 
		id < elements; 
		id += blockDim.x * gridDim.x)
		d_output[id] = d_input[id] + summand;
}