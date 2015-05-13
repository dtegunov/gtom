#include "Prerequisites.h"

TEST(Projection, Backward)
{
	cudaDeviceReset();

	//Case 1:
	{
		int3 dimsvolume = {8, 8, 8};
		int2 dimsimage = {8, 8};
		tfloat3 angles = tfloat3(0.0f, 0.0f, 0.0f);
		tfloat2 shift = tfloat2(0, 0);
		tfloat2 scale = tfloat2(1, 1);
		tfloat weight = (tfloat)1;
		tfloat* d_inputvolume = (tfloat*)CudaMallocValueFilled(Elements(dimsvolume), (tfloat)0);
		tfloat* d_inputimage = (tfloat*)CudaMallocFromBinaryFile("Data\\Projection\\Input_Backward_1.bin");
		tfloat* desired_output = (tfloat*)MallocFromBinaryFile("Data\\Projection\\Output_Backward_1.bin");
		
		d_ProjBackward(d_inputvolume, dimsvolume, tfloat3(0, 0, 0), d_inputimage, dimsimage, &angles, &shift, &scale, T_INTERP_CUBIC, true, 1);

		tfloat* h_output = (tfloat*)MallocFromDeviceArray(d_inputvolume, Elements(dimsvolume) * sizeof(tfloat));
	
		double MeanRelative = GetMeanRelativeError((tfloat*)desired_output, (tfloat*)h_output, Elements(dimsvolume));
		ASSERT_LE(MeanRelative, 1e-5);

		cudaFree(d_inputvolume);
		cudaFree(d_inputimage);
		free(desired_output);
		free(h_output);
	}

	cudaDeviceReset();
}