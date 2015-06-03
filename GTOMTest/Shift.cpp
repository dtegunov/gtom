#include "Prerequisites.h"

TEST(Transformation, Shift)
{
	cudaDeviceReset();

	//Case 1:
	{
		int3 dims = toInt3(8, 8, 1);
		tfloat* h_input = (tfloat*)malloc(Elements(dims) * sizeof(tfloat));
		for (uint i = 0; i < Elements(dims); i++)
			h_input[i] = i;
		tfloat* d_input = (tfloat*)CudaMallocFromHostArray(h_input, Elements(dims) * sizeof(tfloat));
		tcomplex* d_inputft;
		cudaMalloc((void**)&d_inputft, ElementsFFT(dims) * sizeof(tcomplex));
		d_FFTR2C(d_input, d_inputft, 2, dims);

		tfloat3 delta = tfloat3(1, 0, 0);
		d_Shift(d_inputft, d_inputft, dims, &delta);

		d_IFFTC2R(d_inputft, d_input, 2, dims);

		tfloat* h_output = (tfloat*)MallocFromDeviceArray(d_input, Elements(dims) * sizeof(tfloat));
		free(h_output);
	}


	cudaDeviceReset();
}