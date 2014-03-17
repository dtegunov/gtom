#include "../Prerequisites.cuh"
#include "../Functions.cuh"


///////////////////////////////////////////
//Equivalent of TOM's tom_bandpass method//
///////////////////////////////////////////

void d_Bandpass(tfloat* d_input, tfloat* d_output, int3 dims, tfloat low, tfloat high, tfloat smooth, tfloat* d_mask, cufftHandle* planforw, cufftHandle* planback, int batch)
{
	int dimensions = DimensionCount(dims);

	//Prepare mask:

	tfloat* d_localmask;

	if(d_mask == NULL)
	{
		tfloat* d_maskhigh = (tfloat*)CudaMallocValueFilled(Elements(dims), (tfloat)1);

		d_SphereMask(d_maskhigh, d_maskhigh, dims, &high, smooth, (tfloat3*)NULL, 1);

		tfloat* d_maskhighFFT;
		cudaMalloc((void**)&d_maskhighFFT, ElementsFFT(dims) * sizeof(tfloat));
		d_RemapFull2HalfFFT(d_maskhigh, d_maskhighFFT, dims);

		d_localmask = d_maskhighFFT;

		tfloat* d_masklowFFT;
		if(low > 0)
		{
			tfloat* d_masklow = (tfloat*)CudaMallocValueFilled(Elements(dims), (tfloat)1);
			d_SphereMask(d_masklow, d_masklow, dims, &low, smooth, (tfloat3*)NULL, 1);
			cudaMalloc((void**)&d_masklowFFT, ElementsFFT(dims) * sizeof(tfloat));
			d_RemapFull2HalfFFT(d_masklow, d_masklowFFT, dims);
			d_SubtractVector(d_localmask, d_masklowFFT, d_localmask, ElementsFFT(dims), 1);

			cudaFree(d_masklow);
			cudaFree(d_masklowFFT);
		}

		cudaFree(d_maskhigh);
	}
	else
		d_localmask = d_mask;

	//Forward FFT:

	tcomplex* d_inputFFT;
	cudaMalloc((void**)&d_inputFFT, ElementsFFT(dims) * sizeof(tcomplex));

	if(planforw == NULL)
		d_FFTR2C(d_input, d_inputFFT, dimensions, dims, batch);
	else
		d_FFTR2C(d_input, d_inputFFT, planforw);

	//Mask FFT:

	d_ComplexMultiplyByVector(d_inputFFT, d_localmask, d_inputFFT, ElementsFFT(dims), batch);

	//Inverse FFT:

	if(planforw == NULL)
		d_IFFTC2R(d_inputFFT, d_output, dimensions, dims, batch);
	else
		d_IFFTC2R(d_inputFFT, d_output, planback);

	cudaFree(d_inputFFT);

	if(d_mask == NULL)
		cudaFree(d_localmask);
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