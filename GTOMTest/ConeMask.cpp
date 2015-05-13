#include "Prerequisites.h"

TEST(Masking, ConeMask)
{
	cudaDeviceReset();

	//Case 1:
	{
		int3 dims = toInt3(64, 64, 64);
		tfloat* d_volume = CudaMallocValueFilled(ElementsFFT(dims), (tfloat)1);

		d_ConeMaskFT(d_volume, d_volume, dims, make_float3(cos(ToRad(30.0)), sin(ToRad(30.0)), 0), ToRad(10.0));

		d_WriteMRC(d_volume, toInt3(dims.x / 2 + 1, dims.y, dims.z), "d_conemask.mrc");
	}

	cudaDeviceReset();
}