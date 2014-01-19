#include "Prerequisites.h"

TEST(Generics, Pad)
{
	cudaDeviceReset();

	//Case 1:
	{
		int3 inputdims = {6, 6, 1};
		int3 paddeddims = {3, 3, 1};

		tfloat* h_input = (tfloat*)malloc(Elements(inputdims) * sizeof(tfloat));
		for (int i = 0; i < Elements(inputdims); i++)
			h_input[i] = (tfloat)i;
		tfloat* d_input = (tfloat*)CudaMallocFromHostArray(h_input, Elements(inputdims) * sizeof(tfloat));

		tfloat* d_output;
		cudaMalloc((void**)&d_output, Elements(paddeddims) * sizeof(tfloat));

		d_Pad(d_input, d_output, inputdims, paddeddims, T_PAD_MODE::T_PAD_TILE, (tfloat)99, 1);
		tfloat* h_output = (tfloat*)MallocFromDeviceArray(d_output, Elements(paddeddims) * sizeof(tfloat));
	
		//double MeanRelative = GetMeanRelativeError((tfloat*)desired_output, (tfloat*)h_output, newdims.x * newdims.y * newdims.z);
		//ASSERT_LE(MeanRelative, 1e-5);

		cudaFree(d_input);
		cudaFree(d_output);
		free(h_output);
	}

	cudaDeviceReset();
}