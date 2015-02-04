#include "Prerequisites.cuh"
#include "Angles.cuh"
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

void d_Periodogram(tfloat* d_image, int2 dimsimage, int3* d_origins, int norigins, int2 dimsregion, tfloat* d_output)
{
	int2 regions;

	if (norigins == 0)
	{
		regions = toInt2(NextMultipleOf(dimsimage.x, dimsregion.x) / dimsregion.x, NextMultipleOf(dimsimage.y, dimsregion.y) / dimsregion.y);

		tfloat2 shift;
		shift.x = regions.x > 1 ? (tfloat)(dimsimage.x - dimsregion.x) / (tfloat)(regions.x - 1) : (dimsimage.x - dimsregion.x) / 2;
		shift.y = regions.y > 1 ? (tfloat)(dimsimage.y - dimsregion.x) / (tfloat)(regions.y - 1) : (dimsimage.y - dimsregion.y) / 2;
		int3* h_origins = (int3*)malloc(Elements2(regions) * sizeof(int3));

		for (int y = 0; y < regions.y; y++)
			for (int x = 0; x < regions.x; x++)
				h_origins[y * regions.x + x] = toInt3(x * shift.x, y * shift.y, 0);
		d_origins = (int3*)CudaMallocFromHostArray(h_origins, Elements2(regions) * sizeof(int3));
		free(h_origins);
	}
	else
	{
		regions = toInt2(norigins, 1);
	}

	size_t memlimit = 64 << 20;
	int batchsize = min((size_t)Elements2(regions), memlimit / ((size_t)Elements2(dimsregion) * sizeof(tfloat)));

	tfloat* d_extracted;
	cudaMalloc((void**)&d_extracted, batchsize * Elements2(dimsregion) * sizeof(tfloat));
	tcomplex* d_extractedft;
	cudaMalloc((void**)&d_extractedft, batchsize * ElementsFFT2(dimsregion) * sizeof(tcomplex));
	tfloat* d_intermediate;
	cudaMalloc((void**)&d_intermediate, ElementsFFT2(dimsregion) * sizeof(tfloat));
	d_ValueFill(d_output, ElementsFFT2(dimsregion), (tfloat)0);

	for (int b = 0; b < Elements2(regions); b += batchsize)
	{
		int curbatch = min(batchsize, Elements2(regions) - b);

		d_ExtractMany(d_image, d_extracted, toInt3(dimsimage), toInt3(dimsregion), d_origins + b, curbatch);

		d_NormMonolithic(d_extracted, d_extracted, Elements2(dimsregion), T_NORM_MEAN01STD, curbatch);
		d_HammingMask(d_extracted, d_extracted, toInt3(dimsregion), NULL, NULL, curbatch);
		d_FFTR2C(d_extracted, d_extractedft, 2, toInt3(dimsregion), curbatch);
		d_Abs(d_extractedft, d_extracted, curbatch * ElementsFFT2(dimsregion));
		d_MultiplyByVector(d_extracted, d_extracted, d_extracted, curbatch * ElementsFFT2(dimsregion));

		d_ReduceAdd(d_extracted, d_intermediate, ElementsFFT2(dimsregion), curbatch);
		d_AddVector(d_output, d_intermediate, d_output, ElementsFFT2(dimsregion));
	}
	d_RemapHalfFFT2Half(d_output, d_output, toInt3(dimsregion));

	if (norigins == 0)
		cudaFree(d_origins);
	cudaFree(d_intermediate);
	cudaFree(d_extractedft);
	cudaFree(d_extracted);
}