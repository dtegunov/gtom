#include "Prerequisites.h"

TEST(ImageManipulation, BandpassNeat)
{
	cudaDeviceReset();

	//Case 1:
	{
		int3 dims = {256, 256, 1};
		tfloat* d_input = (tfloat*)CudaMallocFromBinaryFile("Data\\ImageManipulation\\Input_BandpassNeat_1.bin");
		//tfloat* desired_output = (tfloat*)MallocFromBinaryFile("Data\\ImageManipulation\\Output_Bandpass_1.bin");
		d_BandpassNeat(d_input, d_input, dims, 5, 512, 0);

		CudaWriteToBinaryFile("Data\\ImageManipulation\\Output_BandpassNeat_1.bin", (void*)d_input, Elements(dims) * sizeof(tfloat));

		//tfloat* h_output = (tfloat*)MallocFromDeviceArray(d_input, Elements(dims) * sizeof(tfloat));
	
		//double MeanRelative = GetMeanRelativeError((tfloat*)desired_output, (tfloat*)h_output, Elements(dims));
		//ASSERT_LE(MeanRelative, 1e-5);

		cudaFree(d_input);
		//free(desired_output);
		//free(h_output);
	}

	cudaDeviceReset();
}