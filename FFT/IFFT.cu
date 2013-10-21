#include "..\Prerequisites.cuh"
#include "..\Functions.cuh"

void d_IFFTC2R(tcomplex* const d_input, tfloat* const d_output, int const ndimensions, int3 const dimensions, int batch)
{
	cufftHandle plan;
	cufftType direction = IS_TFLOAT_DOUBLE ? CUFFT_Z2D : CUFFT_C2R;
	int n[3] = { dimensions.z, dimensions.y, dimensions.x };

	CudaSafeCall((cudaError)cufftPlanMany(&plan, ndimensions, n + (3 - ndimensions),
										  NULL, 1, 0,
										  NULL, 1, 0,
										  direction, batch));

	CudaSafeCall((cudaError)cufftSetCompatibilityMode(plan, CUFFT_COMPATIBILITY_NATIVE));
	#ifdef TOM_DOUBLE
		CudaSafeCall((cudaError)cufftExecZ2D(plan, d_input, d_output));
	#else
		CudaSafeCall((cudaError)cufftExecC2R(plan, d_input, d_output));
	#endif

	CudaSafeCall((cudaError)cufftDestroy(plan));

	size_t elements = dimensions.x * dimensions.y * dimensions.z;
	d_MultiplyByScalar(d_output, d_output, elements, 1.0f / (float)elements);
}

void d_IFFTZ2D(cufftDoubleComplex* const d_input, double* const d_output, int const ndimensions, int3 const dimensions, int batch)
{
	cufftHandle plan;
	cufftType direction = CUFFT_Z2D;
	int n[3] = { dimensions.z, dimensions.y, dimensions.x };

	CudaSafeCall((cudaError)cufftPlanMany(&plan, ndimensions, n + (3 - ndimensions),
										  NULL, 1, 0,
										  NULL, 1, 0,
										  direction, batch));

	CudaSafeCall((cudaError)cufftSetCompatibilityMode(plan, CUFFT_COMPATIBILITY_NATIVE));
	#ifdef TOM_DOUBLE
		CudaSafeCall((cudaError)cufftExecZ2D(plan, d_input, d_output));
	#else
		CudaSafeCall((cudaError)cufftExecZ2D(plan, d_input, d_output));
	#endif

	CudaSafeCall((cudaError)cufftDestroy(plan));

	size_t elements = dimensions.x * dimensions.y * dimensions.z;
	d_MultiplyByScalar(d_output, d_output, elements, 1.0 / (double)elements);
}

void d_IFFTC2RFull(tcomplex* const d_input, tfloat* const d_output, int const ndimensions, int3 const dimensions, int batch)
{
	tcomplex* d_unpadded;
	cudaMalloc((void**)&d_unpadded, (dimensions.x / 2 + 1) * dimensions.y * dimensions.z * sizeof(tcomplex));

	d_HermitianSymmetryTrim(d_input, d_unpadded, dimensions, batch);
	cudaDeviceSynchronize();
	d_IFFTC2R(d_unpadded, d_output, ndimensions, dimensions, batch);

	cudaFree(d_unpadded);
}

void d_IFFTC2C(tcomplex* const d_input, tcomplex* const d_output, int const ndimensions, int3 const dimensions, int batch)
{
	cufftHandle plan;
	cufftType direction = IS_TFLOAT_DOUBLE ? CUFFT_Z2Z : CUFFT_C2C;
	int n[3] = { dimensions.z, dimensions.y, dimensions.x };

	CudaSafeCall((cudaError)cufftPlanMany(&plan, ndimensions, n + (3 - ndimensions),
										  NULL, 1, 0,
										  NULL, 1, 0,
										  direction, batch));

	CudaSafeCall((cudaError)cufftSetCompatibilityMode(plan, CUFFT_COMPATIBILITY_NATIVE));
	#ifdef TOM_DOUBLE
		CudaSafeCall((cudaError)cufftExecZ2Z(plan, d_input, d_output));
	#else
		CudaSafeCall((cudaError)cufftExecC2C(plan, d_input, d_output, CUFFT_INVERSE));
	#endif
	
	CudaSafeCall((cudaError)cufftDestroy(plan));

	size_t elements = dimensions.x * dimensions.y * dimensions.z ;
	d_MultiplyByScalar((tfloat*)d_output, (tfloat*)d_output, elements * 2, 1.0f / (float)elements);
}

void IFFTC2R(tcomplex* const h_input, tfloat* const h_output, int const ndimensions, int3 const dimensions, int batch)
{
	size_t reallength = dimensions.x * dimensions.y * dimensions.z;
	size_t complexlength = (dimensions.x / 2 + 1) * dimensions.y * dimensions.z;

	tcomplex* d_A = (tcomplex*)CudaMallocFromHostArray(h_input, complexlength * sizeof(tcomplex));

	d_IFFTC2R(d_A, (tfloat*)d_A, ndimensions, dimensions, batch);

	cudaMemcpy(h_output, d_A, reallength * sizeof(tfloat), cudaMemcpyDeviceToHost);
	cudaFree(d_A);
}

void IFFTC2RFull(tcomplex* const h_input, tfloat* const h_output, int const ndimensions, int3 const dimensions, int batch)
{
	size_t reallength = dimensions.x * dimensions.y * dimensions.z;
	size_t complexlength = dimensions.x * dimensions.y * dimensions.z;

	tcomplex* d_A = (tcomplex*)CudaMallocFromHostArray(h_input, complexlength * sizeof(tcomplex));
	//tfloat* d_B;
	//cudaMalloc((void**)&d_B, reallength * sizeof(tfloat));

	d_IFFTC2RFull(d_A, (tfloat*)d_A, ndimensions, dimensions, batch);

	cudaMemcpy(h_output, d_A, reallength * sizeof(tfloat), cudaMemcpyDeviceToHost);
	cudaFree(d_A);
	//cudaFree(d_B);
}

void IFFTC2C(tcomplex* const h_input, tcomplex* const h_output, int const ndimensions, int3 const dimensions, int batch)
{
	size_t complexlength = dimensions.x * dimensions.y * dimensions.z;

	tcomplex* d_A = (tcomplex*)CudaMallocFromHostArray(h_input, complexlength * sizeof(tcomplex));

	d_IFFTC2C(d_A, d_A, ndimensions, dimensions, batch);

	cudaMemcpy(h_output, d_A, complexlength * sizeof(tcomplex), cudaMemcpyDeviceToHost);
	cudaFree(d_A);
}