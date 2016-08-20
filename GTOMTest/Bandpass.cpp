#include "Prerequisites.h"

TEST(ImageManipulation, Bandpass)
{
	cudaDeviceReset();

	//Case 1:
	/*{
		int3 dims = {256, 1, 1};
		tfloat* d_input = (tfloat*)CudaMallocFromBinaryFile("Data\\ImageManipulation\\Input_Bandpass_1.bin");
		tfloat* desired_output = (tfloat*)MallocFromBinaryFile("Data\\ImageManipulation\\Output_Bandpass_1.bin");
		d_Bandpass(d_input, d_input, dims, 5, 126, 10);
		tfloat* h_output = (tfloat*)MallocFromDeviceArray(d_input, Elements(dims) * sizeof(tfloat));
	
		double MeanRelative = GetMeanRelativeError((tfloat*)desired_output, (tfloat*)h_output, Elements(dims));
		ASSERT_LE(MeanRelative, 1e-5);

		cudaFree(d_input);
		free(desired_output);
		free(h_output);
	}

	//Case 2:
	{
		int3 dims = {256, 255, 1};
		tfloat* d_input = (tfloat*)CudaMallocFromBinaryFile("Data\\ImageManipulation\\Input_Bandpass_2.bin");
		tfloat* desired_output = (tfloat*)MallocFromBinaryFile("Data\\ImageManipulation\\Output_Bandpass_2.bin");
		d_Bandpass(d_input, d_input, dims, 5, 126, 10);
		tfloat* h_output = (tfloat*)MallocFromDeviceArray(d_input, Elements(dims) * sizeof(tfloat));
	
		double MeanRelative = GetMeanRelativeError((tfloat*)desired_output, (tfloat*)h_output, Elements(dims));
		ASSERT_LE(MeanRelative, 1e-5);

		cudaFree(d_input);
		free(desired_output);
		free(h_output);
	}

	//Case 3:
	{
		int3 dims = {256, 255, 66};
		tfloat* d_input = (tfloat*)CudaMallocFromBinaryFile("Data\\ImageManipulation\\Input_Bandpass_3.bin");
		tfloat* desired_output = (tfloat*)MallocFromBinaryFile("Data\\ImageManipulation\\Output_Bandpass_3.bin");
		d_Bandpass(d_input, d_input, dims, 5, 28, 6);
		tfloat* h_output = (tfloat*)MallocFromDeviceArray(d_input, Elements(dims) * sizeof(tfloat));
	
		double MeanRelative = GetMeanRelativeError((tfloat*)desired_output, (tfloat*)h_output, Elements(dims));
		ASSERT_LE(MeanRelative, 5e-5);

		cudaFree(d_input);
		free(desired_output);
		free(h_output);
	}

	//Case 4:
	{
		int3 dims = {1855, 1855, 1};
		tfloat* d_input = (tfloat*)CudaMallocFromBinaryFile("Data\\ImageManipulation\\Input_Bandpass_4.bin");
		tfloat* desired_output = (tfloat*)MallocFromBinaryFile("Data\\ImageManipulation\\Output_Bandpass_4.bin");
		d_Bandpass(d_input, d_input, dims, 5, 1855 / 5, 20);
		tfloat* h_output = (tfloat*)MallocFromDeviceArray(d_input, Elements(dims) * sizeof(tfloat));
	
		double MeanRelative = GetMeanRelativeError((tfloat*)desired_output, (tfloat*)h_output, Elements(dims));
		ASSERT_LE(MeanRelative, 5e-5);

		cudaFree(d_input);
		free(desired_output);
		free(h_output);
	}*/

	//Case 5:
	{
		int3 dims = toInt3(959, 927, 300);
		tfloat* h_input = MallocValueFilled(Elements(dims), 0.0f);
		h_input[0] = 1.0f;
		tfloat* d_input = (tfloat*)CudaMallocFromHostArray(h_input, Elements(dims) * sizeof(tfloat));
		//d_RemapFullFFT2Full(d_input, d_input, dims);

		d_BandpassNonCubic(d_input, d_input, dims, 0, 0.1f);

		d_WriteMRC(d_input, dims, "d_bandpassnoncubic.mrc");
	}

	cudaDeviceReset();
}