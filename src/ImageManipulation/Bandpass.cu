#include "Prerequisites.cuh"
#include "FFT.cuh"
#include "Generics.cuh"
#include "Helper.cuh"
#include "ImageManipulation.cuh"
#include "Masking.cuh"


///////////////////////////////////////////
//Equivalent of TOM's tom_bandpass method//
///////////////////////////////////////////

void d_Bandpass(tfloat* d_input, tfloat* d_output, int3 dims, tfloat low, tfloat high, tfloat smooth, tfloat* d_mask, cufftHandle* planforw, cufftHandle* planback, int batch)
{
	tcomplex* d_inputft;
	cudaMalloc((void**)&d_inputft, ElementsFFT(dims) * batch * sizeof(tcomplex));

	if (planforw == NULL)
		d_FFTR2C(d_input, d_inputft, DimensionCount(dims), dims, batch);
	else
		d_FFTR2C(d_input, d_inputft, planforw);

	d_Bandpass(d_inputft, d_inputft, dims, low, high, smooth, d_mask, batch);

	if (planback == NULL)
		d_IFFTC2R(d_inputft, d_output, DimensionCount(dims), dims, batch);
	else
		d_IFFTC2R(d_inputft, d_output, planback);

	cudaFree(d_inputft);
}

void d_Bandpass(tcomplex* d_inputft, tcomplex* d_outputft, int3 dims, tfloat low, tfloat high, tfloat smooth, tfloat* d_mask, int batch)
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

	//Mask FFT:

	d_ComplexMultiplyByVector(d_inputft, d_localmask, d_outputft, ElementsFFT(dims), batch);

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
	d_Bandpass(d_paddedinput, d_paddedinput, paddeddims, low * scalefactor, high * scalefactor, smooth * scalefactor, NULL, NULL, NULL, batch);
	d_Pad(d_paddedinput, d_input, paddeddims, dims, T_PAD_MODE::T_PAD_TILE, (tfloat)0, batch);

	cudaFree(d_paddedinput);
}