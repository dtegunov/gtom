#include "Prerequisites.h"

TEST(ImageManipulation, Bandpass)
{
	cudaDeviceReset();

	//Case 1:
	{
		int3 dims = {256, 1, 1};
		tfloat* d_input = (tfloat*)CudaMallocFromBinaryFile("Data\\ImageManipulation\\Input_Bandpass_1.bin");
		tfloat* desired_output = (tfloat*)MallocFromBinaryFile("Data\\ImageManipulation\\Output_Bandpass_1.bin");
		d_Bandpass(d_input, d_input, dims, 5, 126, 10);
		tfloat* h_output = (tfloat*)MallocFromDeviceArray(d_input, dims.x * dims.y * dims.z * sizeof(tfloat));
	
		double MeanRelative = GetMeanRelativeError((tfloat*)desired_output, (tfloat*)h_output, dims.x * dims.y * dims.z);
		ASSERT_LE(MeanRelative, 1e-5);

		cudaFree(d_input);
		free(desired_output);
		free(h_output);
	}

	//Case 2:
	{
		int3 dims = {256, 255, 1};
		tfloat* d_input = (tfloat*)CudaMallocFromBinaryFile("Data\\ImageManipulation\\Input_Bandpass_2.bin");
		tfloat* desired_output = (tfloat*)MallocFromBinaryFile("Data\\ImageManipulation\\Output_Bandpass_2.bin");
		d_Bandpass(d_input, d_input, dims, 5, 126, 10);
		tfloat* h_output = (tfloat*)MallocFromDeviceArray(d_input, dims.x * dims.y * dims.z * sizeof(tfloat));
	
		double MeanRelative = GetMeanRelativeError((tfloat*)desired_output, (tfloat*)h_output, dims.x * dims.y * dims.z);
		ASSERT_LE(MeanRelative, 1e-5);

		cudaFree(d_input);
		free(desired_output);
		free(h_output);
	}

	//Case 3:
	{
		int3 dims = {256, 255, 66};
		tfloat* d_input = (tfloat*)CudaMallocFromBinaryFile("Data\\ImageManipulation\\Input_Bandpass_3.bin");
		tfloat* desired_output = (tfloat*)MallocFromBinaryFile("Data\\ImageManipulation\\Output_Bandpass_3.bin");
		d_Bandpass(d_input, d_input, dims, 5, 28, 6);
		tfloat* h_output = (tfloat*)MallocFromDeviceArray(d_input, dims.x * dims.y * dims.z * sizeof(tfloat));
	
		double MeanRelative = GetMeanRelativeError((tfloat*)desired_output, (tfloat*)h_output, dims.x * dims.y * dims.z);
		ASSERT_LE(MeanRelative, 5e-5);

		cudaFree(d_input);
		free(desired_output);
		free(h_output);
	}

	cudaDeviceReset();
}