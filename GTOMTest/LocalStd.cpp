#include "Prerequisites.h"

TEST(Generics, LocalStd)
{
	cudaDeviceReset();

	//Case 1:
	{
		int3 dimsmap = toInt3(16, 16, 1);

		tfloat* h_input = MallocValueFilled(Elements(dimsmap), (tfloat)1);
		for (int i = 0; i < Elements(dimsmap); i++)
		{
			h_input[i] = i % 2 == 0 ? 1 : 0;
		}

		tfloat* d_input = (tfloat*)CudaMallocFromHostArray(h_input, Elements(dimsmap) * sizeof(tfloat));

		tfloat* d_std = CudaMallocValueFilled(Elements(dimsmap), (tfloat)0);
		tfloat* d_mean = CudaMallocValueFilled(Elements(dimsmap), (tfloat)0);

		d_LocalStd(d_input, dimsmap, 32, d_std, d_mean);

		tfloat* h_std = (tfloat*)MallocFromDeviceArray(d_std, Elements(dimsmap) * sizeof(tfloat));
		tfloat* h_mean = (tfloat*)MallocFromDeviceArray(d_mean, Elements(dimsmap) * sizeof(tfloat));

		free(h_std);
		free(h_mean);
	}


	cudaDeviceReset();
}