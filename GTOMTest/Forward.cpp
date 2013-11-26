#include "Prerequisites.h"

TEST(Projection, Forward)
{
	cudaDeviceReset();

	//Case 1:
	{
		int3 dimsvolume = {8, 8, 8};
		int3 dimsimage = {8, 8, 1};
		tfloat2 angles = tfloat2((tfloat)0 / (tfloat)2, (tfloat)0 / (tfloat)2);
		tfloat weight = (tfloat)1;
		tfloat* d_inputvolume = (tfloat*)CudaMallocFromBinaryFile("Data\\Projection\\Input_Forward_1.bin");
		tfloat* d_inputimage = (tfloat*)CudaMallocValueFilled(Elements(dimsimage), (tfloat)0);
		tfloat* desired_output = (tfloat*)MallocFromBinaryFile("Data\\Projection\\Output_Forward_1.bin");
		
		d_ProjForward(d_inputvolume, dimsvolume, d_inputimage, dimsimage, &angles);

		tfloat* h_output = (tfloat*)MallocFromDeviceArray(d_inputimage, Elements(dimsimage) * sizeof(tfloat));
	
		double MeanAbsolute = GetMeanAbsoluteError((tfloat*)desired_output, (tfloat*)h_output, Elements(dimsimage));
		ASSERT_LE(MeanAbsolute, 1e-5);

		cudaFree(d_inputvolume);
		cudaFree(d_inputimage);
		free(desired_output);
		free(h_output);
	}

	cudaDeviceReset();
}