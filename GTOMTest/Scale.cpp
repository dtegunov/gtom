#include "Prerequisites.h"

TEST(Transformation, Scale)
{
	cudaDeviceReset();

	//Case 1:
	{
		int3 olddims = {4, 1, 1};
		int3 newdims = {4000, 1, 1};
		//tfloat* d_input = (tfloat*)CudaMallocFromBinaryFile("Data\\Transformation\\Input_Shift_1.bin");
		//tfloat* desired_output = (tfloat*)MallocFromBinaryFile("Data\\Transformation\\Output_Shift_1.bin");
		tfloat* h_input = (tfloat*)malloc(olddims.x * sizeof(tfloat));
		for (int i = 0; i < olddims.x; i++)
			h_input[i] = i;
		tfloat* d_input = (tfloat*)CudaMallocFromHostArray(h_input, olddims.x * sizeof(tfloat));
		tfloat* d_output;
		cudaMalloc((void**)&d_output, newdims.x * sizeof(tfloat));

		d_Scale(d_input, d_output, olddims, newdims, T_INTERP_MODE::T_INTERP_CUBIC);
		tfloat* h_output = (tfloat*)MallocFromDeviceArray(d_output, newdims.x * sizeof(tfloat));
	
		//double MeanRelative = GetMeanRelativeError((tfloat*)desired_output, (tfloat*)h_output, dims.x * dims.y * dims.z);
		//ASSERT_LE(MeanRelative, 1e-5);

		cudaFree(d_input);
		cudaFree(d_output);
		//free(desired_output);
		free(h_input);
		free(h_output);
	}


	cudaDeviceReset();
}