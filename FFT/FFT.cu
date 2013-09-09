#include "..\Prerequisites.cuh"
#include "..\Functions.cuh"

__declspec(dllexport) void __stdcall FFT(tfloat* const d_input, tcomplex* const d_output, int const ndimensions, int3 const dimensions)
{
	cufftHandle plan;
	#ifdef TOM_DOUBLE
		#define CUFFT_FORWARD CUFFT_D2Z
	#else
		#define CUFFT_FORWARD CUFFT_R2C
	#endif

	switch (ndimensions)
	{
		case 1:
			CudaSafeCall((cudaError)cufftPlan1d(&plan, dimensions.x, CUFFT_FORWARD, 1));
			break;
		case 2:
			CudaSafeCall((cudaError)cufftPlan2d(&plan, dimensions.x, dimensions.y, CUFFT_FORWARD));
			break;
		case 3:
			CudaSafeCall((cudaError)cufftPlan3d(&plan, dimensions.x, dimensions.y, dimensions.z, CUFFT_FORWARD));
			break;
		default:
			throw;
			break;
	}

	CudaSafeCall((cudaError)cufftSetCompatibilityMode(plan, CUFFT_COMPATIBILITY_NATIVE));
	#ifdef TOM_DOUBLE
		CudaSafeCall((cudaError)cufftExecD2Z(plan, d_input, d_output));
	#else
		CudaSafeCall((cudaError)cufftExecR2C(plan, d_input, d_output));
	#endif
	
	CudaSafeCall((cudaError)cufftDestroy(plan));
}