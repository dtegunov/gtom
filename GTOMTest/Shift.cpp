#include "Prerequisites.h"

TEST(Transformation, Shift)
{
	cudaDeviceReset();

	//Case 1:
	{
		int3 dims = {16, 16, 1};
		tfloat3 shift = tfloat3(1.0f, 2.0f, 0.0f);
		tfloat* d_input = (tfloat*)CudaMallocFromBinaryFile("Data\\Transformation\\Input_Shift_1.bin");
		tfloat* desired_output = (tfloat*)MallocFromBinaryFile("Data\\Transformation\\Output_Shift_1.bin");
		d_Shift(d_input, d_input, dims, &shift, 1);
		tfloat* h_output = (tfloat*)MallocFromDeviceArray(d_input, dims.x * dims.y * dims.z * sizeof(tfloat));
	
		double MeanRelative = GetMeanRelativeError((tfloat*)desired_output, (tfloat*)h_output, dims.x * dims.y * dims.z);
		ASSERT_LE(MeanRelative, 1e-5);

		cudaFree(d_input);
		free(desired_output);
		free(h_output);
	}

	//Case 2:
	{
		int3 dims = {16, 16, 1};
		tfloat3 shift = tfloat3(1.5f, 2.5f, 0.0f);
		tfloat* d_input = (tfloat*)CudaMallocFromBinaryFile("Data\\Transformation\\Input_Shift_2.bin");
		tfloat* desired_output = (tfloat*)MallocFromBinaryFile("Data\\Transformation\\Output_Shift_2.bin");
		d_Shift(d_input, d_input, dims, &shift, 1);
		tfloat* h_output = (tfloat*)MallocFromDeviceArray(d_input, dims.x * dims.y * dims.z * sizeof(tfloat));
	
		double MeanRelative = GetMeanRelativeError((tfloat*)desired_output, (tfloat*)h_output, dims.x * dims.y * dims.z);
		ASSERT_LE(MeanRelative, 1e-5);

		cudaFree(d_input);
		free(desired_output);
		free(h_output);
	}


	cudaDeviceReset();
}