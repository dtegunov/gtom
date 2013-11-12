#include "Prerequisites.h"

TEST(ImageManipulation, Xray)
{
	cudaDeviceReset();

	//Case 1:
	{
		int3 dims = {16, 15, 1};
		tfloat* d_input = (tfloat*)CudaMallocFromBinaryFile("Data\\ImageManipulation\\Input_Xray_1.bin");
		tfloat* desired_output = (tfloat*)MallocFromBinaryFile("Data\\ImageManipulation\\Output_Xray_1.bin");

		d_Xray(d_input, d_input, dims, (tfloat)4.6, 2);
		tfloat* h_output = (tfloat*)MallocFromDeviceArray(d_input, Elements(dims) * sizeof(tfloat));
	
		double MeanRelative = GetMeanRelativeError((tfloat*)desired_output, (tfloat*)h_output, dims.x * dims.y * dims.z);
		ASSERT_LE(MeanRelative, 1e-5);

		cudaFree(d_input);
		free(desired_output);
		free(h_output);
	}

	cudaDeviceReset();
}