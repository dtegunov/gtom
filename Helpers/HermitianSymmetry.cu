#include "../Prerequisites.cuh"
#include "../Functions.cuh"

__global__ void HermitianSymmetryPad2DFirstKernel(tcomplex* const d_input, tcomplex* const d_output, uint3 const dimensions);
__global__ void HermitianSymmetryPad3DFirstKernel(tcomplex* const d_input, tcomplex* const d_output, uint3 const dimensions);
__global__ void HermitianSymmetryPad2DSecondKernel(tcomplex* const d_input, tcomplex* const d_output, uint3 const dimensions);
__global__ void HermitianSymmetryPad3DSecondKernel(tcomplex* const d_input, tcomplex* const d_output, uint3 const dimensions);
__global__ void HermitianSymmetryTrimKernel(tcomplex* const d_input, tcomplex* const d_output, uint3 const dimensions);


////////////////////
//Symmetry Padding//
////////////////////

void d_HermitianSymmetryPad(tcomplex* const d_input, tcomplex* const d_output, int3 const dims, int batch)
{
	size_t elementsFull = Elements(dims);
	size_t elementsTrimmed = ElementsFFT(dims);

	int TpB = 256;
	dim3 grid = dim3((dims.x / 2 + 1 + TpB - 1) / TpB, dims.y, dims.z);
	for (int b = 0; b < batch; b++)
		if(dims.z > 1)
			HermitianSymmetryPad3DFirstKernel <<<grid, TpB>>> (d_input + elementsTrimmed * b, d_output + elementsFull * b, toUint3(dims.x, dims.y, dims.z));
		else
			HermitianSymmetryPad2DFirstKernel <<<grid, TpB>>> (d_input + elementsTrimmed * b, d_output + elementsFull * b, toUint3(dims.x, dims.y, dims.z));
	cudaStreamQuery(0);
	grid = dim3((dims.x / 2 + TpB - 1) / TpB, dims.y, dims.z);
	for (int b = 0; b < batch; b++)
		if(dims.z > 1)
			HermitianSymmetryPad3DSecondKernel <<<grid, TpB>>> (d_input + elementsTrimmed * b, d_output + elementsFull * b, toUint3(dims.x, dims.y, dims.z));
		else
			HermitianSymmetryPad2DSecondKernel <<<grid, TpB>>> (d_input + elementsTrimmed * b, d_output + elementsFull * b, toUint3(dims.x, dims.y, dims.z));
	cudaStreamQuery(0);
}

__global__ void HermitianSymmetryPad2DFirstKernel(tcomplex* const d_input, tcomplex* const d_output, uint3 const dimensions)
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	if(x >= dimensions.x / 2 + 1)
		return;
	uint y = blockIdx.y;

	d_output[y * dimensions.x + x] = d_input[y * (dimensions.x / 2 + 1) + x];
}

__global__ void HermitianSymmetryPad3DFirstKernel(tcomplex* const d_input, tcomplex* const d_output, uint3 const dimensions)
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	if(x >= dimensions.x / 2 + 1)
		return;
	uint y = blockIdx.y;
	uint z = blockIdx.z;

	d_output[(z * dimensions.y + y) * dimensions.x + x] = d_input[(z * dimensions.y + y) * (dimensions.x / 2 + 1) + x];
}

__global__ void HermitianSymmetryPad2DSecondKernel(tcomplex* const d_input, tcomplex* const d_output, uint3 const dimensions)
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x + (dimensions.x / 2 + 1);
	if(x >= dimensions.x)
		return;
	uint y = blockIdx.y;
	
	__shared__ uint warpoutputoffset;
	__shared__ uint warpinputoffset;

	if(threadIdx.x == 0)
	{
		warpoutputoffset = y * dimensions.x;
		warpinputoffset = ((dimensions.y - y) % dimensions.y) * (dimensions.x / 2 + 1);
	}
	__syncthreads();

	tcomplex val = d_input[warpinputoffset + (dimensions.x - x)];
	val.y = -val.y;
	d_output[warpoutputoffset + x] = val;
}

__global__ void HermitianSymmetryPad3DSecondKernel(tcomplex* const d_input, tcomplex* const d_output, uint3 const dimensions)
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x + (dimensions.x / 2 + 1);
	if(x >= dimensions.x)
		return;
	uint y = blockIdx.y;
	uint z = blockIdx.z;
	
	__shared__ uint warpoutputoffset;
	__shared__ uint warpinputoffset;

	if(threadIdx.x == 0)
	{
		warpoutputoffset = (z * dimensions.y + y) * dimensions.x;
		warpinputoffset = (((dimensions.z - z) % dimensions.z) * dimensions.y + ((dimensions.y - y) % dimensions.y)) * (dimensions.x / 2 + 1);
	}
	__syncthreads();

	tcomplex val = d_input[warpinputoffset + (dimensions.x - x)];
	val.y = -val.y;
	d_output[warpoutputoffset + x] = val;
}


/////////////////////
//Symmetry Trimming//
/////////////////////

void d_HermitianSymmetryTrim(tcomplex* const d_input, tcomplex* const d_output, int3 const dims, int batch)
{
	int TpB = 256;
	dim3 grid = dim3(((dims.x / 2 + 1) + TpB - 1) / TpB, dims.y, dims.z);
	size_t elementsFull = Elements(dims);
	size_t elementsTrimmed = ElementsFFT(dims);
	for (int b = 0; b < batch; b++)
		HermitianSymmetryTrimKernel <<<grid, TpB>>> (d_input + elementsFull * b, d_output + elementsTrimmed * b, toUint3(dims.x, dims.y, dims.z));
}

__global__ void HermitianSymmetryTrimKernel(tcomplex* const d_input, tcomplex* const d_output, uint3 const dimensions)
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	if(x >= dimensions.x / 2 + 1)
		return;
	uint y = blockIdx.y;
	uint z = blockIdx.z;
	
	d_output[(z * dimensions.y + y) * (dimensions.x / 2 + 1) + x] = d_input[(z * dimensions.y + y) * dimensions.x + x];
}