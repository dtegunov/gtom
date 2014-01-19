#include "../Prerequisites.cuh"
#include "../Functions.cuh"


///////////////////////////////////////////
//Equivalent of TOM's tom_bandpass method//
///////////////////////////////////////////

void d_Bandpass(tfloat* d_input, tfloat* d_output, int3 dims, tfloat low, tfloat high, tfloat smooth, int batch)
{
	int dimensions = DimensionCount(dims);
	size_t elements = dims.x * dims.y * dims.z;
	size_t elementsFFT = (dims.x / 2 + 1) * dims.y * dims.z;

	//Prepare mask:

	tfloat* d_maskhigh = (tfloat*)CudaMallocValueFilled(elements, (tfloat)1);

	d_SphereMask(d_maskhigh, d_maskhigh, dims, &high, smooth, (tfloat3*)NULL, 1);

	tfloat* d_maskhighFFT;
	cudaMalloc((void**)&d_maskhighFFT, elementsFFT * sizeof(tfloat));
	d_RemapFull2HalfFFT(d_maskhigh, d_maskhighFFT, dims);

	tfloat* d_mask = d_maskhighFFT;

	tfloat* d_masklowFFT;
	if(low > 0)
	{
		tfloat* d_masklow = (tfloat*)CudaMallocValueFilled(elements, (tfloat)1);
		d_SphereMask(d_masklow, d_masklow, dims, &low, smooth, (tfloat3*)NULL, 1);
		cudaMalloc((void**)&d_masklowFFT, elementsFFT * sizeof(tfloat));
		d_RemapFull2HalfFFT(d_masklow, d_masklowFFT, dims);
		d_SubtractVector(d_mask, d_masklowFFT, d_mask, elementsFFT, 1);

		cudaFree(d_masklow);
		cudaFree(d_masklowFFT);
	}

	cudaFree(d_maskhigh);

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


//////////////////////////////////////////////////////////////////////////////////////////////////
//Same as d_Bandpass, but mirror-pads the input to double size to avoid artefacts at the borders//
//////////////////////////////////////////////////////////////////////////////////////////////////

void d_BandpassNeat(tfloat* d_input, tfloat* d_output, int3 dims, tfloat low, tfloat high, tfloat smooth, int batch)
{
	int dimensions = DimensionCount(dims);
	int3 paddeddims = toInt3(NextPow2(dims.x * 2), dimensions > 1 ? NextPow2(dims.y * 2) : 1, dimensions > 2 ? NextPow2(dims.z * 2) : 1);
	tfloat scalefactor = max((tfloat)paddeddims.x / (tfloat)dims.x, max((tfloat)paddeddims.y / (tfloat)dims.y, (tfloat)paddeddims.z / (tfloat)dims.z));

	tfloat* d_paddedinput;
	cudaMalloc((void**)&d_paddedinput, Elements(paddeddims) * batch * sizeof(tfloat));

	d_Pad(d_input, d_paddedinput, dims, paddeddims, T_PAD_MODE::T_PAD_MIRROR, (tfloat)0, batch);
	d_Bandpass(d_paddedinput, d_paddedinput, paddeddims, low * scalefactor, high * scalefactor, smooth * scalefactor, batch);
	d_Pad(d_paddedinput, d_input, paddeddims, dims, T_PAD_MODE::T_PAD_TILE, (tfloat)0, batch);

	cudaFree(d_paddedinput);
}