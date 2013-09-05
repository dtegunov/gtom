#include "..\Prerequisites.cuh"
#include "..\Functions.cuh"

__declspec(dllexport) void __stdcall FFT(cufftReal* const d_input, cufftComplex* const d_output, int const ndimensions, int3 const dimensions)
{
	cufftHandle plan;

	switch (ndimensions)
	{
		case 1:
			CudaSafeCall((cudaError)cufftPlan1d(&plan, dimensions.x, CUFFT_R2C, 1));
			break;
		case 2:
			CudaSafeCall((cudaError)cufftPlan2d(&plan, dimensions.x, dimensions.y, CUFFT_R2C));
			break;
		case 3:
			CudaSafeCall((cudaError)cufftPlan3d(&plan, dimensions.x, dimensions.y, dimensions.z, CUFFT_R2C));
			break;
		default:
			throw;
			break;
	}

	CudaSafeCall((cudaError)cufftSetCompatibilityMode(plan, CUFFT_COMPATIBILITY_NATIVE));
	CudaSafeCall((cudaError)cufftExecR2C(plan, d_input, d_output));
	
	CudaSafeCall((cudaError)cufftDestroy(plan));
}