#include "../Prerequisites.cuh"
#include "../Functions.cuh"

__global__ void HermitianSymmetryPad2DFirstKernel(tcomplex* d_input, tcomplex* d_output, uint3 dimensions, size_t elementsTrimmed, size_t elementsFull);
__global__ void HermitianSymmetryPad3DFirstKernel(tcomplex* d_input, tcomplex* d_output, uint3 dimensions, size_t elementsTrimmed, size_t elementsFull);
__global__ void HermitianSymmetryPad2DSecondKernel(tcomplex* d_input, tcomplex* d_output, uint3 dimensions, size_t elementsTrimmed, size_t elementsFull);
__global__ void HermitianSymmetryPad3DSecondKernel(tcomplex* d_input, tcomplex* d_output, uint3 dimensions, size_t elementsTrimmed, size_t elementsFull);
__global__ void HermitianSymmetryTrimKernel(tcomplex* d_input, tcomplex* d_output, uint3 dimensions, size_t elementsTrimmed, size_t elementsFull);


////////////////////
//Symmetry Padding//
////////////////////

void d_HermitianSymmetryPad(tcomplex* const d_input, tcomplex* const d_output, int3 const dims, int batch)
{
	size_t elementsFull = Elements(dims);
	size_t elementsTrimmed = ElementsFFT(dims);

	int TpB = min(128, NextMultipleOf(dims.x / 2 + 1, 32));
	dim3 grid = dim3(dims.y, dims.z, batch);
	if(dims.z > 1)
		HermitianSymmetryPad3DFirstKernel <<<grid, TpB>>> (d_input, d_output, toUint3(dims.x, dims.y, dims.z), elementsTrimmed, elementsFull);
	else
		HermitianSymmetryPad2DFirstKernel <<<grid, TpB>>> (d_input, d_output, toUint3(dims.x, dims.y, dims.z), elementsTrimmed, elementsFull);
	cudaStreamQuery(0);

	TpB = min(128, NextMultipleOf(dims.x / 2, 32));
	grid = dim3(dims.y, dims.z, batch);
	if(dims.z > 1)
		HermitianSymmetryPad3DSecondKernel <<<grid, TpB>>> (d_input, d_output, toUint3(dims.x, dims.y, dims.z), elementsTrimmed, elementsFull);
	else
		HermitianSymmetryPad2DSecondKernel <<<grid, TpB>>> (d_input, d_output, toUint3(dims.x, dims.y, dims.z), elementsTrimmed, elementsFull);
	cudaStreamQuery(0);
}

__global__ void HermitianSymmetryPad2DFirstKernel(tcomplex* d_input, tcomplex* d_output, uint3 dimensions, size_t elementsTrimmed, size_t elementsFull)
{
	d_input += elementsTrimmed * blockIdx.z;
	d_output += elementsFull * blockIdx.z;

	for(uint x = threadIdx.x; x < elementsTrimmed; x += blockDim.x)
		d_output[blockIdx.x * dimensions.x + x] = d_input[blockIdx.x * (dimensions.x / 2 + 1) + x];
}

__global__ void HermitianSymmetryPad3DFirstKernel(tcomplex* d_input, tcomplex* d_output, uint3 dimensions, size_t elementsTrimmed, size_t elementsFull)
{
	d_input += elementsTrimmed * blockIdx.z;
	d_output += elementsFull * blockIdx.z;

	for(uint x = threadIdx.x; x < elementsTrimmed; x += blockDim.x)
	{
		uint y = blockIdx.x;
		uint z = blockIdx.y;

		d_output[(z * dimensions.y + y) * dimensions.x + x] = d_input[(z * dimensions.y + y) * (dimensions.x / 2 + 1) + x];
	}
}

__global__ void HermitianSymmetryPad2DSecondKernel(tcomplex* d_input, tcomplex* d_output, uint3 dimensions, size_t elementsTrimmed, size_t elementsFull)
{
	d_input += elementsTrimmed * blockIdx.z;
	d_output += elementsFull * blockIdx.z;

	for(uint x = threadIdx.x + (dimensions.x / 2 + 1); x < dimensions.x; x += blockDim.x)
	{
		uint y = blockIdx.x;

		tcomplex val = d_input[((dimensions.y - y) % dimensions.y) * (dimensions.x / 2 + 1) + (dimensions.x - x)];
		val.y = -val.y;
		d_output[y * dimensions.x + x] = val;
	}
}

__global__ void HermitianSymmetryPad3DSecondKernel(tcomplex* d_input, tcomplex* d_output, uint3 dimensions, size_t elementsTrimmed, size_t elementsFull)
{
	d_input += elementsTrimmed * blockIdx.z;
	d_output += elementsFull * blockIdx.z;

	for(uint x = threadIdx.x + (dimensions.x / 2 + 1); x < dimensions.x; x += blockDim.x)
	{
		uint y = blockIdx.x;
		uint z = blockIdx.y;

		tcomplex val = d_input[(((dimensions.z - z) % dimensions.z) * dimensions.y + ((dimensions.y - y) % dimensions.y)) * (dimensions.x / 2 + 1) + (dimensions.x - x)];
		val.y = -val.y;
		d_output[(z * dimensions.y + y) * dimensions.x + x] = val;
	}
}


/////////////////////
//Symmetry Trimming//
/////////////////////

void d_HermitianSymmetryTrim(tcomplex* const d_input, tcomplex* const d_output, int3 const dims, int batch)
{
	int TpB = min(256, NextMultipleOf(dims.x / 2 + 1, 32));
	dim3 grid = dim3(dims.y, dims.z, batch);
	size_t elementsFull = Elements(dims);
	size_t elementsTrimmed = ElementsFFT(dims);

	HermitianSymmetryTrimKernel <<<grid, TpB>>> (d_input, d_output, toUint3(dims.x, dims.y, dims.z), elementsTrimmed, elementsFull);
}

__global__ void HermitianSymmetryTrimKernel(tcomplex* d_input, tcomplex* d_output, uint3 dimensions, size_t elementsTrimmed, size_t elementsFull)
{
	d_input += elementsFull * blockIdx.z;
	d_output += elementsTrimmed * blockIdx.z;

	for(uint x = threadIdx.x; x < dimensions.x / 2 + 1; x += blockDim.x)
	{
		uint y = blockIdx.x;
		uint z = blockIdx.y;
	
		d_output[(z * dimensions.y + y) * (dimensions.x / 2 + 1) + x] = d_input[(z * dimensions.y + y) * dimensions.x + x];
	}
}