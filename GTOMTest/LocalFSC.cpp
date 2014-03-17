#include "Prerequisites.h"

TEST(Resolution, LocalFSC)
{
	cudaDeviceReset();

	//Case 1:
	{
		int3 dims = {280, 280, 280};
		int shells = 20;
		tfloat* d_input1 = (tfloat*)CudaMallocFromBinaryFile("Data\\Resolution\\Input1_LocalFSC_1.bin");
		tfloat* d_input2 = (tfloat*)CudaMallocFromBinaryFile("Data\\Resolution\\Input2_LocalFSC_1.bin");
		tfloat* d_resolution = (tfloat*)CudaMallocValueFilled(Elements(dims), (tfloat)0);
		
		d_LocalFSC(d_input1, d_input2, dims, d_resolution, 40, shells, (tfloat)0.143);

		tfloat* h_resolution = (tfloat*)MallocFromDeviceArray(d_resolution, Elements(dims) * sizeof(tfloat));
		free(h_resolution);

		/*tfloat* h_output = (tfloat*)MallocFromDeviceArray(d_input, dims.x * dims.y * dims.z * sizeof(tfloat));
	
		double MeanRelative = GetMeanRelativeError((tfloat*)desired_output, (tfloat*)h_output, dims.x * dims.y * dims.z);
		ASSERT_LE(MeanRelative, 1e-5);*/
		
		cudaFree(d_resolution);
		cudaFree(d_input2);
		cudaFree(d_input1);
	}

	cudaDeviceReset();
}