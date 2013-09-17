#include "..\Prerequisites.cuh"
#include "..\Functions.cuh"

__global__ void HermitianSymmetryPadKernel(tcomplex* const d_input, tcomplex* const d_output, int3 const dimensions);
__global__ void HermitianSymmetryTrimKernel(tcomplex* const d_input, tcomplex* const d_output, int3 const dimensions);


////////////////////
//Symmetry Padding//
////////////////////

void d_HermitianSymmetryPad(tcomplex* const d_input, tcomplex* const d_output, int3 const dimensions)
{
	int TpB = 256;
	dim3 grid = dim3((dimensions.x + TpB - 1) / TpB, dimensions.y, dimensions.z);
	HermitianSymmetryPadKernel <<<grid, TpB>>> (d_input, d_output, dimensions);

	CudaSafeCall(cudaDeviceSynchronize());
}

__global__ void HermitianSymmetryPadKernel(tcomplex* const d_input, tcomplex* const d_output, int3 const dimensions)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if(x >= dimensions.x)
		return;
	int y = blockIdx.y;
	int z = blockIdx.z;
	
	size_t outputaddress = (z * dimensions.y + y) * dimensions.x + x;

	bool conjugate = false;
	if(x >= dimensions.x / 2 + 1)
	{
		conjugate = true;
		x = dimensions.x - x;
		if(y > 0)
			y = dimensions.y - y;
		if(z > 0)
			z = dimensions.z - z;
	}

	size_t inputaddress = (z * dimensions.y + y) * (dimensions.x / 2 + 1) + x;
	if(!conjugate)
		d_output[outputaddress] = d_input[inputaddress];
	else
	{
		tcomplex conj = d_input[inputaddress];
		conj.y = -conj.y;
		d_output[outputaddress] = conj;
	}
}


/////////////////////
//Symmetry Trimming//
/////////////////////

void d_HermitianSymmetryTrim(tcomplex* const d_input, tcomplex* const d_output, int3 const dimensions)
{
	int TpB = 256;
	dim3 grid = dim3(((dimensions.x / 2 + 1) + TpB - 1) / TpB, dimensions.y, dimensions.z);
	HermitianSymmetryTrimKernel <<<grid, TpB>>> (d_input, d_output, dimensions);

	CudaSafeCall(cudaDeviceSynchronize());
}

__global__ void HermitianSymmetryTrimKernel(tcomplex* const d_input, tcomplex* const d_output, int3 const dimensions)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if(x >= dimensions.x / 2 + 1)
		return;
	int y = blockIdx.y;
	int z = blockIdx.z;
	
	size_t inputaddress = (z * dimensions.y + y) * dimensions.x + x;
	size_t outputaddress = (z * dimensions.y + y) * (dimensions.x / 2 + 1) + x;
	d_output[outputaddress] = d_input[inputaddress];
}