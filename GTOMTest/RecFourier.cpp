#include "Prerequisites.h"

TEST(Reconstruction, Fourier)
{
	cudaDeviceReset();

	//Case 1:
	{
		int3 dimsvolume = {16, 16, 16};
		int3 dimsimage = {16, 16, 2249};

		tfloat* d_inputproj = (tfloat*)CudaMallocFromBinaryFile("Data\\Reconstruction\\Input_ARTProj_2.bin");
		tfloat2* h_inputangles = (tfloat2*)MallocFromBinaryFile("Data\\Reconstruction\\Input_ARTAngles_2.bin");
		tfloat* desired_output = (tfloat*)MallocFromBinaryFile("Data\\Reconstruction\\Output_ART_2.bin");

		tfloat* d_volume;
		cudaMalloc((void**)&d_volume, Elements(dimsvolume) * sizeof(tfloat));
		
		d_ReconstructFourier(d_inputproj, dimsimage, d_volume, dimsvolume, h_inputangles);

		tfloat* h_output = (tfloat*)MallocFromDeviceArray(d_volume, Elements(dimsvolume) * sizeof(tfloat));
		tfloat outputmax = (tfloat)-999999;
		for (int i = 0; i < Elements(dimsvolume); i++)
			outputmax = max(outputmax, h_output[i]);
		//for (int i = 0; i < Elements(dimsvolume); i++)
			//h_output[i] /= outputmax;
	
		double MeanAbsolute = GetMeanAbsoluteError((tfloat*)desired_output, (tfloat*)h_output, Elements(dimsvolume));
		//ASSERT_LE(MeanRelative, 1e-5);

		cudaFree(d_volume);
		cudaFree(d_inputproj);
		free(desired_output);
		free(h_output);
	}

	cudaDeviceReset();
}