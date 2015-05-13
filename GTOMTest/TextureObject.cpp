#include "Prerequisites.h"

TEST(Helpers, TextureObject)
{
	cudaDeviceReset();

	//Case 1:
	{
		int3 dims = toInt3(512, 512, 512);
		tfloat* d_dummy = CudaMallocValueFilled(Elements(dims), (tfloat)1);
		cudaArray_t a;
		cudaTex t;

		d_BindTextureTo3DArray(d_dummy, a, t, dims, cudaFilterModeLinear, false);

		cudaDestroyTextureObject(t);
		cudaFreeArray(a);
		cudaFree(d_dummy);
	}

	cudaDeviceReset();
}