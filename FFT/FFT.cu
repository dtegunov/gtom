#include "..\Prerequisites.cuh"
#include "..\Functions.cuh"

void d_FFTR2C(tfloat* const d_input, tcomplex* const d_output, int const ndimensions, int3 const dimensions, int batch)
{
	cufftHandle plan;
	cufftType direction = IS_TFLOAT_DOUBLE ? CUFFT_D2Z : CUFFT_R2C;
	int n[3] = { dimensions.z, dimensions.y, dimensions.x };

	CudaSafeCall((cudaError)cufftPlanMany(&plan, ndimensions, n + (3 - ndimensions),
										  NULL, 1, 0,
										  NULL, 1, 0,
										  direction, batch));

	CudaSafeCall((cudaError)cufftSetCompatibilityMode(plan, CUFFT_COMPATIBILITY_NATIVE));
	#ifdef TOM_DOUBLE
		CudaSafeCall((cudaError)cufftExecD2Z(plan, d_input, d_output));
	#else
		CudaSafeCall((cudaError)cufftExecR2C(plan, d_input, d_output));
	#endif
	
	cudaDeviceSynchronize();
	
	CudaSafeCall((cudaError)cufftDestroy(plan));
}

void d_FFTR2CFull(tfloat* const d_input, tcomplex* const d_output, int const ndimensions, int3 const dimensions, int batch)
{
	tcomplex* d_unpadded;
	cudaMalloc((void**)&d_unpadded, (dimensions.x / 2 + 1) * dimensions.y * dimensions.z * batch * sizeof(tcomplex));

	d_FFTR2C(d_input, d_unpadded, ndimensions, dimensions, batch);
	d_HermitianSymmetryPad(d_unpadded, d_output, dimensions, batch);

	cudaFree(d_unpadded);
}

void d_FFTC2C(tcomplex* const d_input, tcomplex* const d_output, int const ndimensions, int3 const dimensions, int batch)
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
		CudaSafeCall((cudaError)cufftExecC2C(plan, d_input, d_output, CUFFT_FORWARD));
	#endif
	
	CudaSafeCall((cudaError)cufftDestroy(plan));
}

void FFTR2C(tfloat* const h_input, tcomplex* const h_output, int const ndimensions, int3 const dimensions, int batch)
{
	size_t reallength = dimensions.x * dimensions.y * dimensions.z;
	size_t complexlength = (dimensions.x / 2 + 1) * dimensions.y * dimensions.z;

	tfloat* d_A;
	cudaMalloc((void**)&d_A, complexlength * batch * sizeof(tcomplex));
	for (int b = 0; b < batch; b++)
		cudaMemcpy(d_A + complexlength * 2 * b, h_input + reallength * b, reallength * sizeof(tfloat), cudaMemcpyHostToDevice);

	d_FFTR2C(d_A, (tcomplex*)d_A, ndimensions, dimensions, batch);

	cudaMemcpy(h_output, d_A, complexlength * batch * sizeof(tcomplex), cudaMemcpyDeviceToHost);
	cudaFree(d_A);
}

void FFTR2CFull(tfloat* const h_input, tcomplex* const h_output, int const ndimensions, int3 const dimensions, int batch)
{
	size_t reallength = dimensions.x * dimensions.y * dimensions.z;
	size_t complexlength = dimensions.x * dimensions.y * dimensions.z;

	tfloat* d_A;
	cudaMalloc((void**)&d_A, complexlength * batch * sizeof(tcomplex));
	for (int b = 0; b < batch; b++)
		cudaMemcpy(d_A + complexlength * 2 * b, h_input + reallength * b, reallength * sizeof(tfloat), cudaMemcpyHostToDevice);

	d_FFTR2CFull(d_A, (tcomplex*)d_A, ndimensions, dimensions, batch);

	cudaMemcpy(h_output, d_A, reallength * batch * sizeof(tcomplex), cudaMemcpyDeviceToHost);
	cudaFree(d_A);
}

void FFTC2C(tcomplex* const h_input, tcomplex* const h_output, int const ndimensions, int3 const dimensions, int batch)
{
	size_t complexlength = dimensions.x * dimensions.y * dimensions.z;

	tcomplex* d_A = (tcomplex*)CudaMallocFromHostArray(h_input, complexlength * batch * sizeof(tcomplex));

	d_FFTC2C(d_A, d_A, ndimensions, dimensions, batch);

	cudaMemcpy(h_output, d_A, complexlength * batch * sizeof(tcomplex), cudaMemcpyDeviceToHost);
	cudaFree(d_A);
}