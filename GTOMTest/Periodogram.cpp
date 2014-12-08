#include "Prerequisites.h"

TEST(CTF, Periodogram)
{
	cudaDeviceReset();

	//Case 1:
	{
		int2 dimsimage = toInt2(2048, 2048);
		int2 dimsregion = toInt2(512, 512);

		tfloat* d_input = (tfloat*)CudaMallocFromBinaryFile("Data/CTF/Input_Periodogram.bin");
		tfloat* d_output = CudaMallocValueFilled(ElementsFFT2(dimsregion), (tfloat)0);

		d_Periodogram(d_input, dimsimage, NULL, 0, dimsregion, d_output);
		CudaWriteToBinaryFile("d_periodogram.bin", d_output, ElementsFFT2(dimsregion) * sizeof(tfloat));
	}

	cudaDeviceReset();
}