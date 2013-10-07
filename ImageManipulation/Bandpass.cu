#include "..\Prerequisites.cuh"
#include "..\Functions.cuh"


///////////////////////////////////////////
//Equivalent of TOM's tom_bandpass method//
///////////////////////////////////////////

void d_Bandpass(tfloat* d_input, tfloat* d_output, int3 dims, tfloat low, tfloat high, tfloat smooth, int batch)
{
	int dimensions = 3 - max(2 - dims.z, 0) - max(2 - dims.y, 0);
	size_t elements = dims.x * dims.y * dims.z;
	size_t elementsFFT = (dims.x / 2 + 1) * dims.y * dims.z;

	//Prepare mask:

	tfloat* d_maskhigh = (tfloat*)CudaMallocValueFilled(elements, (tfloat)1);
	tfloat* d_masklow = (tfloat*)CudaMallocValueFilled(elements, (tfloat)1);

	d_SphereMask(d_maskhigh, d_maskhigh, dims, &high, smooth, (tfloat3*)NULL, 1);
	d_SphereMask(d_masklow, d_masklow, dims, &low, smooth, (tfloat3*)NULL, 1);

	tfloat* d_maskhighFFT;
	cudaMalloc((void**)&d_maskhighFFT, elementsFFT * sizeof(tfloat));
	tfloat* d_masklowFFT;
	cudaMalloc((void**)&d_masklowFFT, elementsFFT * sizeof(tfloat));

	d_RemapFull2HalfFFT(d_maskhigh, d_maskhighFFT, dims);
	d_RemapFull2HalfFFT(d_masklow, d_masklowFFT, dims);

	tfloat* d_mask = d_maskhighFFT;
	d_SubtractVector(d_mask, d_masklowFFT, d_mask, elementsFFT, 1);

	cudaFree(d_maskhigh);
	cudaFree(d_masklow);
	cudaFree(d_masklowFFT);

	//Forward FFT:

	tcomplex* d_inputFFT;
	cudaMalloc((void**)&d_inputFFT, elementsFFT * sizeof(tcomplex));

	d_FFTR2C(d_input, d_inputFFT, dimensions, dims, batch);

	//Mask FFT:

	d_ComplexMultiplyByVector(d_inputFFT, d_mask, d_inputFFT, elementsFFT, batch);

	//Inverse FFT:

	d_IFFTC2R(d_inputFFT, d_output, dimensions, dims, batch);

	cudaFree(d_inputFFT);
	cudaFree(d_mask);
}