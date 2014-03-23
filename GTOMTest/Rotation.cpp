#include "Prerequisites.h"

TEST(Transformation, Rotation)
{
	cudaDeviceReset();

	//Case 1:
	{
		int3 dims = {8, 8, 8};
		tfloat3 angles = tfloat3(ToRad(0.0f), ToRad(180.0f), ToRad(0.0f));
		tfloat* d_input = (tfloat*)CudaMallocFromBinaryFile("Data\\Transformation\\Input_Rotate3D_1.bin");
		tfloat* desired_output = (tfloat*)MallocFromBinaryFile("Data\\Transformation\\Output_Rotate3D_1.bin");
		tfloat* d_output = CudaMallocValueFilled(Elements(dims), (tfloat)-1);

		d_Rotate3D(d_input, d_output, dims, &angles, T_INTERP_MODE::T_INTERP_CUBIC);
		tfloat* h_output = (tfloat*)MallocFromDeviceArray(d_output, Elements(dims) * sizeof(tfloat));
	
		double MeanRelative = GetMeanRelativeError((tfloat*)desired_output, (tfloat*)h_output, Elements(dims));
		//ASSERT_LE(MeanRelative, 1e-5);

		cudaFree(d_input);
		cudaFree(d_output);
		free(desired_output);
		free(h_output);
	}


	cudaDeviceReset();
}