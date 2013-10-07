#include "Prerequisites.h"

TEST(Masking, RectangleMask)
{
	cudaDeviceReset();

	//Case 1:
	{
		int3 dims = {256, 1, 1};
		int3 rectsize = { 126, 0, 0 };
		tfloat* d_input = (tfloat*)CudaMallocFromBinaryFile("Data\\Masking\\Input_Rectmask_1.bin");
		tfloat* desired_output = (tfloat*)MallocFromBinaryFile("Data\\Masking\\Output_Rectmask_1.bin");
		d_RectangleMask(d_input, d_input, dims, rectsize, (tfloat)0, (int3*)NULL, 1);
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
		int3 rectsize = { 126, 126, 0 };
		tfloat* d_input = (tfloat*)CudaMallocFromBinaryFile("Data\\Masking\\Input_Rectmask_2.bin");
		tfloat* desired_output = (tfloat*)MallocFromBinaryFile("Data\\Masking\\Output_Rectmask_2.bin");
		d_RectangleMask(d_input, d_input, dims, rectsize, (tfloat)0, (int3*)NULL, 1);
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
		int3 rectsize = { 3, 5, 6 };
		tfloat* d_input = (tfloat*)CudaMallocFromBinaryFile("Data\\Masking\\Input_Rectmask_3.bin");
		tfloat* desired_output = (tfloat*)MallocFromBinaryFile("Data\\Masking\\Output_Rectmask_3.bin");
		d_RectangleMask(d_input, d_input, dims, rectsize, (tfloat)0, (int3*)NULL, 1);
		tfloat* h_output = (tfloat*)MallocFromDeviceArray(d_input, dims.x * dims.y * dims.z * sizeof(tfloat));
	
		double MeanRelative = GetMeanRelativeError((tfloat*)desired_output, (tfloat*)h_output, dims.x * dims.y * dims.z);
		ASSERT_LE(MeanRelative, 1e-5);

		cudaFree(d_input);
		free(desired_output);
		free(h_output);
	}

	//Case 4:
	{
		int3 dims = {256, 256, 1};
		int3 rectsize = { 17, 16, 1 };
		tfloat* d_input = (tfloat*)CudaMallocFromBinaryFile("Data\\Masking\\Input_Rectmask_4.bin");
		tfloat* desired_output = (tfloat*)MallocFromBinaryFile("Data\\Masking\\Output_Rectmask_4.bin");
		d_RectangleMask(d_input, d_input, dims, rectsize, (tfloat)10, (int3*)NULL, 1);
		tfloat* h_output = (tfloat*)MallocFromDeviceArray(d_input, dims.x * dims.y * dims.z * sizeof(tfloat));
	
		double MeanRelative = GetMeanRelativeError((tfloat*)desired_output, (tfloat*)h_output, dims.x * dims.y * dims.z);
		ASSERT_LE(MeanRelative, 1e-5);

		cudaFree(d_input);
		free(desired_output);
		free(h_output);
	}

	cudaDeviceReset();
}