#include "Prerequisites.h"

TEST(Projection, Backward)
{
	cudaDeviceReset();

	//Case 1:
	{
		int3 dimsvolume = {8, 8, 8};
		int3 dimsimage = {8, 8, 1};
		tfloat2 angles = tfloat2((tfloat)0.0, (tfloat)0.0);
		tfloat weight = (tfloat)1;
		tfloat* d_inputvolume = (tfloat*)CudaMallocValueFilled(Elements(dimsvolume), (tfloat)0);
		tfloat* d_inputimage = (tfloat*)CudaMallocFromBinaryFile("Data\\Projection\\Input_Backward_1.bin");
		tfloat* desired_output = (tfloat*)MallocFromBinaryFile("Data\\Projection\\Output_Backward_1.bin");
		
		d_ProjBackward(d_inputvolume, dimsvolume, d_inputimage, dimsimage, &angles, &weight);

		tfloat* h_output = (tfloat*)MallocFromDeviceArray(d_inputvolume, Elements(dimsvolume) * sizeof(tfloat));
	
		double MeanRelative = GetMeanRelativeError((tfloat*)desired_output, (tfloat*)h_output, Elements(dimsvolume));
		ASSERT_LE(MeanRelative, 1e-5);

		cudaFree(d_inputvolume);
		cudaFree(d_inputimage);
		free(desired_output);
		free(h_output);
	}

	//Case 2:
	{
		int3 dimsvolume = {8, 8, 8};
		int3 dimsimage = {8, 8, 1};
		tfloat2 angles = tfloat2((tfloat)PI / (tfloat)2, (tfloat)0.0);
		tfloat weight = (tfloat)1;
		tfloat* d_inputvolume = (tfloat*)CudaMallocValueFilled(Elements(dimsvolume), (tfloat)0);
		tfloat* d_inputimage = (tfloat*)CudaMallocFromBinaryFile("Data\\Projection\\Input_Backward_2.bin");
		tfloat* desired_output = (tfloat*)MallocFromBinaryFile("Data\\Projection\\Output_Backward_2.bin");
		
		d_ProjBackward(d_inputvolume, dimsvolume, d_inputimage, dimsimage, &angles, &weight);

		tfloat* h_output = (tfloat*)MallocFromDeviceArray(d_inputvolume, Elements(dimsvolume) * sizeof(tfloat));
	
		double MeanRelative = GetMeanRelativeError((tfloat*)desired_output, (tfloat*)h_output, Elements(dimsvolume));
		ASSERT_LE(MeanRelative, 1e-5);

		cudaFree(d_inputvolume);
		cudaFree(d_inputimage);
		free(desired_output);
		free(h_output);
	}

	//Case 3:
	{
		int3 dimsvolume = {8, 8, 8};
		int3 dimsimage = {8, 8, 1};
		tfloat2 angles = tfloat2((tfloat)0.0, (tfloat)PI / (tfloat)2);
		tfloat weight = (tfloat)1;
		tfloat* d_inputvolume = (tfloat*)CudaMallocValueFilled(Elements(dimsvolume), (tfloat)0);
		tfloat* d_inputimage = (tfloat*)CudaMallocFromBinaryFile("Data\\Projection\\Input_Backward_3.bin");
		tfloat* desired_output = (tfloat*)MallocFromBinaryFile("Data\\Projection\\Output_Backward_3.bin");
		
		d_ProjBackward(d_inputvolume, dimsvolume, d_inputimage, dimsimage, &angles, &weight);

		tfloat* h_output = (tfloat*)MallocFromDeviceArray(d_inputvolume, Elements(dimsvolume) * sizeof(tfloat));
	
		double MeanRelative = GetMeanRelativeError((tfloat*)desired_output, (tfloat*)h_output, Elements(dimsvolume));
		ASSERT_LE(MeanRelative, 1e-1);

		cudaFree(d_inputvolume);
		cudaFree(d_inputimage);
		free(desired_output);
		free(h_output);
	}

	//Case 4:
	{
		int3 dimsvolume = {8, 8, 8};
		int3 dimsimage = {8, 8, 1};
		tfloat2 angles = tfloat2((tfloat)PI / (tfloat)6, (tfloat)PI / (tfloat)4);
		tfloat weight = (tfloat)1;
		tfloat* d_inputvolume = (tfloat*)CudaMallocValueFilled(Elements(dimsvolume), (tfloat)0);
		tfloat* d_inputimage = (tfloat*)CudaMallocFromBinaryFile("Data\\Projection\\Input_Backward_4.bin");
		tfloat* desired_output = (tfloat*)MallocFromBinaryFile("Data\\Projection\\Output_Backward_4.bin");
		
		d_ProjBackward(d_inputvolume, dimsvolume, d_inputimage, dimsimage, &angles, &weight);

		tfloat* h_output = (tfloat*)MallocFromDeviceArray(d_inputvolume, Elements(dimsvolume) * sizeof(tfloat));
	
		double MeanAbsolute = GetMeanAbsoluteError((tfloat*)desired_output, (tfloat*)h_output, Elements(dimsvolume));
		ASSERT_LE(MeanAbsolute, 5e-2);

		cudaFree(d_inputvolume);
		cudaFree(d_inputimage);
		free(desired_output);
		free(h_output);
	}

	cudaDeviceReset();
}