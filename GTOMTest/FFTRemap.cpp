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
		d_RemapFull2HalfFFT(d_input, d_output, dims);
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
		d_RemapFull2HalfFFT(d_input, d_output, dims);
		tfloat* h_output = (tfloat*)MallocFromDeviceArray(d_output, (dims.x / 2 + 1) * dims.y * dims.z * sizeof(tfloat));
	
		double MeanRelative = GetMeanRelativeError((tfloat*)desired_output, (tfloat*)h_output, (dims.x / 2 + 1) * dims.y * dims.z);
		ASSERT_LE(MeanRelative, 1e-5);

		cudaFree(d_input);
		free(desired_output);
		free(h_output);
	}

	//Case 3:
	{
<<<<<<< HEAD
		int3 dims = {8, 8, 8};
		tfloat* d_input = (tfloat*)CudaMallocFromBinaryFile("Data\\FFT\\Input_Remap_3.bin");
		tfloat* h_input = (tfloat*)MallocFromDeviceArray(d_input, (dims.x / 2 + 1) * dims.y * dims.z * sizeof(tfloat));
		tfloat* d_output;
		cudaMalloc((void**)&d_output, (dims.x / 2 + 1) * dims.y * dims.z * sizeof(tfloat));
		tfloat* desired_output = (tfloat*)MallocFromBinaryFile("Data\\FFT\\Output_Remap_3.bin");
		d_RemapHalfToHalfFFT(d_input, d_output, dims);
		tfloat* h_output = (tfloat*)MallocFromDeviceArray(d_output, (dims.x / 2 + 1) * dims.y * dims.z * sizeof(tfloat));
	
		double MeanRelative = GetMeanRelativeError((tfloat*)desired_output, (tfloat*)h_output, (dims.x / 2 + 1) * dims.y * dims.z);
=======
		int3 dims = {8, 8, 1};
		tfloat* d_input = (tfloat*)CudaMallocFromBinaryFile("Data\\FFT\\Input_Remap_3.bin");
		tfloat* d_output;
		cudaMalloc((void**)&d_output, dims.x * dims.y * dims.z * sizeof(tfloat));
		tfloat* desired_output = (tfloat*)MallocFromBinaryFile("Data\\FFT\\Output_Remap_3.bin");
		d_RemapFullFFT2Full(d_input, d_output, dims);
		tfloat* h_output = (tfloat*)MallocFromDeviceArray(d_output, dims.x * dims.y * dims.z * sizeof(tfloat));
	
		double MeanRelative = GetMeanRelativeError((tfloat*)desired_output, (tfloat*)h_output, dims.x * dims.y * dims.z);
>>>>>>> 77ee24d2625debc91b0cc36e1f8bdad326e7221b
		ASSERT_LE(MeanRelative, 1e-5);

		cudaFree(d_input);
		free(desired_output);
		free(h_output);
<<<<<<< HEAD
		free(h_input);
=======
>>>>>>> 77ee24d2625debc91b0cc36e1f8bdad326e7221b
	}

	//Case 4:
	{
<<<<<<< HEAD
		int3 dims = {8, 8, 8};
		tfloat* d_input = (tfloat*)CudaMallocFromBinaryFile("Data\\FFT\\Input_Remap_4.bin");
		tfloat* h_input = (tfloat*)MallocFromDeviceArray(d_input, (dims.x / 2 + 1) * dims.y * dims.z * sizeof(tfloat));
		tfloat* d_output;
		cudaMalloc((void**)&d_output, (dims.x / 2 + 1) * dims.y * dims.z * sizeof(tfloat));
		tfloat* desired_output = (tfloat*)MallocFromBinaryFile("Data\\FFT\\Output_Remap_4.bin");
		d_RemapHalfFFTToHalf(d_input, d_output, dims);
		tfloat* h_output = (tfloat*)MallocFromDeviceArray(d_output, (dims.x / 2 + 1) * dims.y * dims.z * sizeof(tfloat));
	
		double MeanRelative = GetMeanRelativeError((tfloat*)desired_output, (tfloat*)h_output, (dims.x / 2 + 1) * dims.y * dims.z);
=======
		int3 dims = {7, 7, 7};
		tfloat* d_input = (tfloat*)CudaMallocFromBinaryFile("Data\\FFT\\Input_Remap_4.bin");
		tfloat* d_output;
		cudaMalloc((void**)&d_output, dims.x * dims.y * dims.z * sizeof(tfloat));
		tfloat* desired_output = (tfloat*)MallocFromBinaryFile("Data\\FFT\\Output_Remap_4.bin");
		d_RemapFullFFT2Full(d_input, d_output, dims);
		tfloat* h_output = (tfloat*)MallocFromDeviceArray(d_output, dims.x * dims.y * dims.z * sizeof(tfloat));
	
		double MeanRelative = GetMeanRelativeError((tfloat*)desired_output, (tfloat*)h_output, dims.x * dims.y * dims.z);
>>>>>>> 77ee24d2625debc91b0cc36e1f8bdad326e7221b
		ASSERT_LE(MeanRelative, 1e-5);

		cudaFree(d_input);
		free(desired_output);
		free(h_output);
<<<<<<< HEAD
		free(h_input);
=======
>>>>>>> 77ee24d2625debc91b0cc36e1f8bdad326e7221b
	}

	cudaDeviceReset();
}