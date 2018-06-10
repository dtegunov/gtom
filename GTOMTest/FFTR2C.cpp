#include "Prerequisites.h"

#include "Prerequisites.h"

TEST(FFT, Forward)
{
	cudaDeviceReset();

	//Case 1:
	{
		int3 dims = toInt3(8, 8, 8);
		tfloat* h_input = MallocValueFilled(Elements(dims), (tfloat)0);
		for (uint i = 0; i < Elements(dims); i++)
			h_input[i] = i % Elements2(dims);

		tfloat* d_input = (tfloat*)CudaMallocFromHostArray(h_input, Elements(dims) * sizeof(tfloat));
		tcomplex* d_output;
		cudaMalloc((void**)&d_output, ElementsFFT(dims) * sizeof(tcomplex));

		d_FFTR2C(d_input, d_output, 2, toInt3(dims.x, dims.y, 1), dims.z);

		tcomplex* h_output = (tcomplex*)MallocFromDeviceArray(d_output, ElementsFFT(dims) * sizeof(tcomplex));
		cout << h_output[0].x << " " << h_output[0].y;
		free(h_output);
	}

	cudaDeviceReset();
}
