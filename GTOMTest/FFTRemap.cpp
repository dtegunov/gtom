#include "Prerequisites.h"

TEST(FFT, FFTRemap)
{
	cudaDeviceReset();

	//Case 1:
	{
		int3 dims = {4, 3, 5};
		tfloat* d_input = (tfloat*)CudaMallocFromBinaryFile("Data\\FFT\\Input_Remap_1.bin");
		tfloat* d_output;
		cudaMalloc((void**)&d_output, (dims.x / 2 + 1) * dims.y * dims.z * sizeof(tfloat));
		tfloat* desired_output = (tfloat*)MallocFromBinaryFile("Data\\FFT\\Output_Remap_1.bin");
		d_RemapFullToHalfFFT(d_input, d_output, dims);
		tfloat* h_output = (tfloat*)MallocFromDeviceArray(d_output, (dims.x / 2 + 1) * dims.y * dims.z * sizeof(tfloat));
	
		double MeanRelative = GetMeanRelativeError((tfloat*)desired_output, (tfloat*)h_output, (dims.x / 2 + 1) * dims.y * dims.z);
		ASSERT_LE(MeanRelative, 1e-5);

		cudaFree(d_input);
		free(desired_output);
		free(h_output);
	}

	//Case 2:
	{
		int3 dims = {8, 8, 8};
		tfloat* d_input = (tfloat*)CudaMallocFromBinaryFile("Data\\FFT\\Input_Remap_2.bin");
		tfloat* d_output;
		cudaMalloc((void**)&d_output, (dims.x / 2 + 1) * dims.y * dims.z * sizeof(tfloat));
		tfloat* desired_output = (tfloat*)MallocFromBinaryFile("Data\\FFT\\Output_Remap_2.bin");
		d_RemapFullToHalfFFT(d_input, d_output, dims);
		tfloat* h_output = (tfloat*)MallocFromDeviceArray(d_output, (dims.x / 2 + 1) * dims.y * dims.z * sizeof(tfloat));
	
		double MeanRelative = GetMeanRelativeError((tfloat*)desired_output, (tfloat*)h_output, (dims.x / 2 + 1) * dims.y * dims.z);
		ASSERT_LE(MeanRelative, 1e-5);

		cudaFree(d_input);
		free(desired_output);
		free(h_output);
	}

	cudaDeviceReset();
}