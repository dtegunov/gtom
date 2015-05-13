#include "Prerequisites.h"

#include "Prerequisites.h"

TEST(FFT, Forward)
{
	cudaDeviceReset();

	//Case 1:
	{
		int2 dims = toInt2(8, 8);
		tfloat* h_input = MallocValueFilled(Elements2(dims), (tfloat)0);
		for (uint i = 0; i < Elements2(dims); i++)
			h_input[i] = i + 1;

		tfloat* d_input = (tfloat*)CudaMallocFromHostArray(h_input, Elements2(dims) * sizeof(tfloat));
		tcomplex* d_output;
		cudaMalloc((void**)&d_output, ElementsFFT2(dims) * sizeof(tcomplex));

		d_FFTR2C(d_input, d_output, 2, toInt3(dims));

		tcomplex* h_output = (tcomplex*)MallocFromDeviceArray(d_output, ElementsFFT2(dims) * sizeof(tcomplex));
		free(h_output);
	}

	cudaDeviceReset();
}