#include "Prerequisites.h"

TEST(Masking, SphereMask)
{
	cudaDeviceReset();

	//Case 1:
	{
		int3 dims = {256, 1, 1};
		tfloat radius = 126;
		tfloat* d_input = (tfloat*)CudaMallocFromBinaryFile("Data\\Masking\\Input_Spheremask_1.bin");
		tfloat* desired_output = (tfloat*)MallocFromBinaryFile("Data\\Masking\\Output_Spheremask_1.bin");
		d_SphereMask(d_input, d_input, dims, &radius, 10, (tfloat3*)NULL);
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
		tfloat radius = 126;
		tfloat* d_input = (tfloat*)CudaMallocFromBinaryFile("Data\\Masking\\Input_Spheremask_2.bin");
		tfloat* desired_output = (tfloat*)MallocFromBinaryFile("Data\\Masking\\Output_Spheremask_2.bin");
		d_SphereMask(d_input, d_input, dims, &radius, 10, (tfloat3*)NULL);
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
		tfloat radius = 3;
		tfloat* d_input = (tfloat*)CudaMallocFromBinaryFile("Data\\Masking\\Input_Spheremask_3.bin");
		tfloat* desired_output = (tfloat*)MallocFromBinaryFile("Data\\Masking\\Output_Spheremask_3.bin");
		d_SphereMask(d_input, d_input, dims, &radius, 2, (tfloat3*)NULL);
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
		tfloat radius = 128;
		tfloat* d_input = (tfloat*)CudaMallocFromBinaryFile("Data\\Masking\\Input_Spheremask_4.bin");
		tfloat* desired_output = (tfloat*)MallocFromBinaryFile("Data\\Masking\\Output_Spheremask_4.bin");
		d_SphereMask(d_input, d_input, dims, &radius, 10, (tfloat3*)NULL);
		tfloat* h_output = (tfloat*)MallocFromDeviceArray(d_input, dims.x * dims.y * dims.z * sizeof(tfloat));
	
		double MeanRelative = GetMeanRelativeError((tfloat*)desired_output, (tfloat*)h_output, dims.x * dims.y * dims.z);
		ASSERT_LE(MeanRelative, 1e-5);

		cudaFree(d_input);
		free(desired_output);
		free(h_output);
	}

	cudaDeviceReset();
}