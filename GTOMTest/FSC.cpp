#include "Prerequisites.h"

TEST(Resolution, FSC)
{
	cudaDeviceReset();

	//Case 1:
	{
		int3 dims = {140, 140, 140};
		int shells = 70;
		tfloat* d_input1 = (tfloat*)CudaMallocFromBinaryFile("Data\\Resolution\\Input1_FSC_1.bin");
		tfloat* d_input2 = (tfloat*)CudaMallocFromBinaryFile("Data\\Resolution\\Input2_FSC_1.bin");
		tfloat* d_curve = (tfloat*)CudaMallocValueFilled(shells * 2, (tfloat)0);
		tfloat* d_resolution = (tfloat*)CudaMallocValueFilled(2, (tfloat)-1);
		tfloat* desired_output = (tfloat*)MallocFromBinaryFile("Data\\Resolution\\Output_FSC_1.bin");
		
		d_FSC(d_input1, d_input2, dims, d_curve, shells, NULL, 2);
		//d_AnisotropicFSC(d_input1, d_input2, dims, d_curve, shells, tfloat3(1, 0, 0), ToRad(90), (tfloat)0);
		//d_FirstIndexOf(d_curve, d_resolution, shells, (tfloat)0.143, T_INTERP_LINEAR, 2);

		tfloat* h_curve = (tfloat*)MallocFromDeviceArray(d_curve, shells * 2 * sizeof(tfloat));
		tfloat* h_resolution = (tfloat*)MallocFromDeviceArray(d_resolution, 2 * sizeof(tfloat));
		free(h_curve);
		free(h_resolution);

		/*tfloat* h_output = (tfloat*)MallocFromDeviceArray(d_input, dims.x * dims.y * dims.z * sizeof(tfloat));
	
		double MeanRelative = GetMeanRelativeError((tfloat*)desired_output, (tfloat*)h_output, dims.x * dims.y * dims.z);
		ASSERT_LE(MeanRelative, 1e-5);*/
		
		cudaFree(d_input1);
		cudaFree(d_input2);
		free(desired_output);
	}

	cudaDeviceReset();
}