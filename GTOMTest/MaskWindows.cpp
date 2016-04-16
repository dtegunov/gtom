#include "Prerequisites.h"

TEST(Masking, Windows)
{
	cudaDeviceReset();

	//Case 1:
	{
		int3 dims = toInt3(1024, 1024, 1);

		tfloat* d_input = CudaMallocValueFilled(Elements(dims), 1.0f);
		d_HammingMaskBorderDistance(d_input, d_input, dims, dims.x / 4);
		d_WriteMRC(d_input, dims, "Mask_Windows.mrc");
	}

	cudaDeviceReset();
}