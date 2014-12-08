#include "Prerequisites.h"

TEST(Generics, Extract)
{
	cudaDeviceReset();

	//Case 1:
	{
		int3 sourcedims = {10, 1, 1};
		int3 regiondims = {5, 1, 1};

		tfloat* h_input = (tfloat*)malloc(Elements(sourcedims) * sizeof(tfloat));
		for (int i = 0; i < Elements(sourcedims); i++)
			h_input[i] = (tfloat)i;
		tfloat* d_input = (tfloat*)CudaMallocFromHostArray(h_input, Elements(sourcedims) * sizeof(tfloat));

		tfloat* d_output;
		cudaMalloc((void**)&d_output, Elements(regiondims) * sizeof(tfloat));

		d_Extract(d_input, d_output, sourcedims, regiondims, toInt3(5, 0, 0));
		tfloat* h_output = (tfloat*)MallocFromDeviceArray(d_output, Elements(regiondims) * sizeof(tfloat));
	
		//double MeanRelative = GetMeanRelativeError((tfloat*)desired_output, (tfloat*)h_output, newdims.x * newdims.y * newdims.z);
		//ASSERT_LE(MeanRelative, 1e-5);

		cudaFree(d_input);
		cudaFree(d_output);
		free(h_output);
	}

	//Case 2:
	{
		int2 sourcedims = {8, 8};
		int2 regiondims = {8, 8};

		tfloat* h_input = (tfloat*)malloc(Elements2(sourcedims) * sizeof(tfloat));
		for (int i = 0; i < Elements2(sourcedims); i++)
			h_input[i] = (tfloat)i;
		tfloat* d_input = (tfloat*)CudaMallocFromHostArray(h_input, Elements2(sourcedims) * sizeof(tfloat));

		tfloat* d_output;
		cudaMalloc((void**)&d_output, Elements2(regiondims) * sizeof(tfloat));

		tfloat2 scale(1, 1);
		tfloat rotation = PI / (tfloat)1;
		tfloat2 translation(sourcedims.x / 2, sourcedims.y / 2);

		d_Extract2DTransformed(d_input, d_output, sourcedims, regiondims, &scale, &rotation, &translation, T_INTERP_MODE::T_INTERP_LINEAR);
		tfloat* h_output = (tfloat*)MallocFromDeviceArray(d_output, Elements2(regiondims) * sizeof(tfloat));
	
		//double MeanRelative = GetMeanRelativeError((tfloat*)desired_output, (tfloat*)h_output, newdims.x * newdims.y * newdims.z);
		//ASSERT_LE(MeanRelative, 1e-5);

		cudaFree(d_input);
		cudaFree(d_output);
		free(h_output);
	}


	cudaDeviceReset();
}