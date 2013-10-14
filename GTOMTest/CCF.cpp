#include "Prerequisites.h"

TEST(Correlation, CCF)
{
	cudaDeviceReset();

	//Case 1:
	{
		int3 dims = {16, 16, 1};
		tfloat* d_input1 = (tfloat*)CudaMallocFromBinaryFile("Data\\Correlation\\Input1_CCF_1.bin");
		tfloat* d_input2 = (tfloat*)CudaMallocFromBinaryFile("Data\\Correlation\\Input2_CCF_1.bin");
		tfloat* desired_output = (tfloat*)MallocFromBinaryFile("Data\\Correlation\\Output_CCF_1.bin");
		d_CCF(d_input1, d_input2, d_input1, dims, false, (tfloat*)NULL);
		tfloat* h_output = (tfloat*)MallocFromDeviceArray(d_input1, dims.x * dims.y * dims.z * sizeof(tfloat));
	
		double MeanRelative = GetMeanRelativeError((tfloat*)desired_output, (tfloat*)h_output, dims.x * dims.y * dims.z);
		ASSERT_LE(MeanRelative, 1e-5);

		cudaFree(d_input1);
		cudaFree(d_input2);
		free(desired_output);
		free(h_output);
	}

	//Case 2:
	{
		int3 dims = {17, 17, 17};
		tfloat* d_input1 = (tfloat*)CudaMallocFromBinaryFile("Data\\Correlation\\Input1_CCF_2.bin");
		tfloat* d_input2 = (tfloat*)CudaMallocFromBinaryFile("Data\\Correlation\\Input2_CCF_2.bin");
		tfloat* desired_output = (tfloat*)MallocFromBinaryFile("Data\\Correlation\\Output_CCF_2.bin");
		d_CCF(d_input1, d_input2, d_input1, dims, true, (tfloat*)NULL);
		tfloat* h_output = (tfloat*)MallocFromDeviceArray(d_input1, dims.x * dims.y * dims.z * sizeof(tfloat));
	
		double MeanRelative = GetMeanRelativeError((tfloat*)desired_output, (tfloat*)h_output, dims.x * dims.y * dims.z);
		ASSERT_LE(MeanRelative, 5e-5);

		cudaFree(d_input1);
		cudaFree(d_input2);
		free(desired_output);
		free(h_output);
	}

	cudaDeviceReset();
}