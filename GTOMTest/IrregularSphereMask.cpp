#include "Prerequisites.h"

TEST(Masking, IrregularSphereMask)
{
	cudaDeviceReset();

	//Case 1:
	{
		int3 dims = {8, 8, 8};
		int2 anglesteps = toInt2(16, 4);
		tfloat* d_input = CudaMallocValueFilled(Elements(dims), (tfloat)1);
		tfloat* d_radiusmap = CudaMallocValueFilled(anglesteps.x * anglesteps.y, (tfloat)3);

		d_IrregularSphereMask(d_input, d_input, dims, d_radiusmap, anglesteps, 0, NULL);

		tfloat* h_input = (tfloat*)MallocFromDeviceArray(d_input, Elements(dims) * sizeof(tfloat));
		free(h_input);

		cudaFree(d_input);
		cudaFree(d_radiusmap);
	}

	cudaDeviceReset();
}