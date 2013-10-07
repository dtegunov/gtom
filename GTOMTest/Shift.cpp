#include "Prerequisites.h"

TEST(ImageManipulation, Shift)
{
	cudaDeviceReset();

	//Case 1:
	{
		int3 dims = {16, 16, 1};
		tfloat* d_input = (tfloat*)CudaMallocFromBinaryFile("Data\\ImageManipulation\\Input_Shift_1.bin");
		tfloat* desired_output = (tfloat*)MallocFromBinaryFile("Data\\ImageManipulation\\Output_Shift_1.bin");
		d_Shift(d_input, d_input, dims, tfloat3(1.0f, 2.0f, 0.0f), 1);
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
		tfloat* d_input = (tfloat*)CudaMallocFromBinaryFile("Data\\ImageManipulation\\Input_Shift_2.bin");
		tfloat* desired_output = (tfloat*)MallocFromBinaryFile("Data\\ImageManipulation\\Output_Shift_2.bin");
		d_Shift(d_input, d_input, dims, tfloat3(1.5f, 2.5f, 0.0f), 1);
		tfloat* h_output = (tfloat*)MallocFromDeviceArray(d_input, dims.x * dims.y * dims.z * sizeof(tfloat));
	
		double MeanRelative = GetMeanRelativeError((tfloat*)desired_output, (tfloat*)h_output, dims.x * dims.y * dims.z);
		ASSERT_LE(MeanRelative, 1e-5);

		cudaFree(d_input);
		free(desired_output);
		free(h_output);
	}

	//Case 3:
	/*{
		int3 dims = {16, 16, 1};
		tfloat* d_input = (tfloat*)CudaMallocFromBinaryFile("Data\\ImageManipulation\\Input_Shift_3.bin");
		tfloat* desired_output = (tfloat*)MallocFromBinaryFile("Data\\ImageManipulation\\Output_Shift_3.bin");
		d_Shift(d_input, d_input, dims, tfloat3(-1.5f, 2.5f, 0.0f), 1);
		tfloat* h_output = (tfloat*)MallocFromDeviceArray(d_input, dims.x * dims.y * dims.z * sizeof(tfloat));
	
		double MeanRelative = GetMeanRelativeError((tfloat*)desired_output, (tfloat*)h_output, dims.x * dims.y * dims.z);
		ASSERT_LE(MeanRelative, 1e-5);

		cudaFree(d_input);
		free(desired_output);
		free(h_output);
	}*/

	cudaDeviceReset();
}