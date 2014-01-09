#include "Prerequisites.h"

TEST(Correlation, LocalPeaks)
{
	cudaDeviceReset();

	//Case 1:
	{
		int3 dims = {8, 8, 1};
		int upscalefactor = 8;
		int3 updims = {1769, 1769, 1};

		tfloat* h_input = MallocValueFilled(Elements(dims), (tfloat)0);
		h_input[1 * dims.x + 1] = (tfloat)1;
		h_input[6 * dims.x + 3] = (tfloat)1;
		tfloat* d_input = (tfloat*)CudaMallocFromHostArray(h_input, Elements(dims) * sizeof(tfloat));

		tfloat* d_upscaled;
		cudaMalloc((void**)&d_upscaled, Elements(updims) * sizeof(tfloat));

		d_Scale(d_input, d_upscaled, dims, updims, T_INTERP_MODE::T_INTERP_FOURIER);
		cudaFree(d_input);
		d_Norm(d_upscaled, d_upscaled, Elements(updims), (int*)NULL, T_NORM_MODE::T_NORM_MEAN01STD, 0);

		tfloat* h_upscaled = (tfloat*)MallocFromDeviceArray(d_upscaled, Elements(updims) * sizeof(tfloat));

		int3* h_peaks;
		int h_peakcount;

		d_LocalPeaks(d_upscaled, &h_peaks, &h_peakcount, updims, 100, (tfloat)4.6);

		cudaFree(d_upscaled);
	}

	cudaDeviceReset();
}