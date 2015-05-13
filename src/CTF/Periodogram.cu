#include "Prerequisites.cuh"
#include "Angles.cuh"
#include "CTF.cuh"
#include "CubicInterp.cuh"
#include "DeviceFunctions.cuh"
#include "FFT.cuh"
#include "Generics.cuh"
#include "Helper.cuh"
#include "ImageManipulation.cuh"
#include "Masking.cuh"
#include "Transformation.cuh"

//////////////////////////////////////////////////////
//Calculate power spectrum based on multiple regions//
//////////////////////////////////////////////////////

void d_CTFPeriodogram(tfloat* d_image, int2 dimsimage, float overlapfraction, int2 dimsregion, tfloat* d_output2d)
{
	// Create uniform grid over the image
	int2 regions;
	int3* h_origins = GetEqualGridSpacing(dimsimage, dimsregion, overlapfraction, regions);
	int3* d_origins = (int3*)CudaMallocFromHostArray(h_origins, Elements2(regions) * sizeof(int3));
	free(h_origins);

	int norigins = Elements2(regions);

	tfloat* d_temp2d;
	cudaMalloc((void**)&d_temp2d, ElementsFFT2(dimsregion) * norigins * sizeof(tfloat));

	// Call the custom-grid version to extract 2D spectra
	d_CTFPeriodogram(d_image, dimsimage, d_origins, norigins, dimsregion, d_temp2d);

	d_ReduceMean(d_temp2d, d_output2d, ElementsFFT2(dimsregion), norigins);

	cudaFree(d_temp2d);
	cudaFree(d_origins);
}

void d_CTFPeriodogram(tfloat* d_image, int2 dimsimage, int3* d_origins, int norigins, int2 dimsregion, tfloat* d_output2d)
{
	size_t memlimit = 256 << 20;
	int batchsize = min((size_t)norigins, memlimit / ((size_t)Elements2(dimsregion) * sizeof(tfloat)));

	tfloat* d_extracted;
	cudaMalloc((void**)&d_extracted, batchsize * Elements2(dimsregion) * sizeof(tfloat));
	tcomplex* d_extractedft;
	cudaMalloc((void**)&d_extractedft, batchsize * ElementsFFT2(dimsregion) * sizeof(tcomplex));

	for (int b = 0; b < norigins; b += batchsize)
	{
		int curbatch = min(batchsize, norigins - b);

		d_ExtractMany(d_image, d_extracted, toInt3(dimsimage), toInt3(dimsregion), d_origins + b, curbatch);
		//d_WriteMRC(d_extracted, toInt3(dimsregion.x, dimsregion.y, curbatch), "d_extracted.mrc");

		d_NormMonolithic(d_extracted, d_extracted, Elements2(dimsregion), T_NORM_MEAN01STD, curbatch);
		d_HammingMask(d_extracted, d_extracted, toInt3(dimsregion), NULL, NULL, curbatch);
		d_NormMonolithic(d_extracted, d_extracted, Elements2(dimsregion), T_NORM_MEAN01STD, curbatch);
		d_FFTR2C(d_extracted, d_extractedft, 2, toInt3(dimsregion), curbatch);
		d_Abs(d_extractedft, d_extracted, curbatch * ElementsFFT2(dimsregion));
		d_AddScalar(d_extracted, d_extracted, curbatch * ElementsFFT2(dimsregion), (tfloat)1e-6);
		d_Log(d_extracted, d_extracted, curbatch * ElementsFFT2(dimsregion));
		d_MultiplyByVector(d_extracted, d_extracted, d_extracted, ElementsFFT2(dimsregion) * curbatch);
		//d_NormMonolithic(d_extracted, d_extracted, ElementsFFT2(dimsregion), T_NORM_MEAN01STD, curbatch);
		d_RemapHalfFFT2Half(d_extracted, d_output2d + b * ElementsFFT2(dimsregion), toInt3(dimsregion), curbatch);
	}

	cudaFree(d_extractedft);
	cudaFree(d_extracted);
}