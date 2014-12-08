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
		regions = toInt2(ceil((tfloat)dimsimage.x / (tfloat)dimsregion.x) * 2, ceil((tfloat)dimsimage.y / (tfloat)dimsregion.y) * 2);

		tfloat2 shift = tfloat2((tfloat)(dimsimage.x - dimsregion.x) / (tfloat)(regions.x - 1), (tfloat)(dimsimage.y - dimsregion.x) / (tfloat)(regions.y - 1));
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

	tfloat* d_extracted;
	cudaMalloc((void**)&d_extracted, Elements2(regions) * Elements2(dimsregion) * sizeof(tfloat));
	tcomplex* d_extractedft;
	cudaMalloc((void**)&d_extractedft, Elements2(regions) * ElementsFFT2(dimsregion) * sizeof(tcomplex));

	d_Extract(d_image, d_extracted, toInt3(dimsimage), toInt3(dimsregion), d_origins, Elements2(regions));

	if (norigins == 0)
		cudaFree(d_origins);

	d_NormMonolithic(d_extracted, d_extracted, Elements2(dimsregion), T_NORM_MEAN01STD, Elements2(regions));
	d_HammingMask(d_extracted, d_extracted, toInt3(dimsregion), NULL, NULL, Elements2(regions));
	d_FFTR2C(d_extracted, d_extractedft, 2, toInt3(dimsregion), Elements2(regions));
	d_Abs(d_extractedft, d_extracted, Elements2(regions) * ElementsFFT2(dimsregion));
	d_MultiplyByVector(d_extracted, d_extracted, d_extracted, Elements2(regions) * Elements2(dimsregion));

	d_ReduceAdd(d_extracted, d_output, ElementsFFT2(dimsregion), Elements2(regions));
	d_RemapHalfFFT2Half(d_output, d_output, toInt3(dimsregion));

	//CudaWriteToBinaryFile("d_output.bin", d_output, ElementsFFT2(dimsregion) * sizeof(tfloat));

	cudaFree(d_extractedft);
	cudaFree(d_extracted);
}