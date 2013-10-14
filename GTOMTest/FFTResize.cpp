#include "Prerequisites.h"

TEST(FFT, FFTResize)
{
	cudaDeviceReset();

	//Case 1:
	{
		int3 olddims = {15, 15, 1};
		int3 newdims = {7, 7, 1};

		tfloat* d_input = (tfloat*)CudaMallocFromBinaryFile("Data\\FFT\\Input_Resize_1.bin");
		tfloat* desired_output = (tfloat*)MallocFromBinaryFile("Data\\FFT\\Output_Resize_1.bin");
		
		tcomplex* d_output;
		cudaMalloc((void**)&d_output, (newdims.x / 2 + 1) * newdims.y * newdims.z * sizeof(tcomplex));
		tcomplex* d_intermediate;
		cudaMalloc((void**)&d_intermediate, (olddims.x / 2 + 1) * olddims.y * olddims.z * sizeof(tcomplex));

		d_FFTR2C(d_input, d_intermediate, DimensionCount(olddims), olddims);
		d_FFTCrop(d_intermediate, d_output, olddims, newdims);

		tcomplex* h_output = (tcomplex*)MallocFromDeviceArray(d_output, (newdims.x / 2 + 1) * newdims.y * newdims.z * sizeof(tcomplex));
	
		double MeanRelative = GetMeanRelativeError((tfloat*)desired_output, (tfloat*)h_output, (newdims.x / 2 + 1) * newdims.y * newdims.z);
		ASSERT_LE(MeanRelative, 1e-5);

		cudaFree(d_input);
		free(desired_output);
		free(h_output);
	}

	//Case 2:
	{
		int3 olddims = {8, 8, 1};
		int3 newdims = {16, 16, 1};

		tfloat* d_input = (tfloat*)CudaMallocFromBinaryFile("Data\\FFT\\Input_Resize_2.bin");
		tfloat* desired_output = (tfloat*)MallocFromBinaryFile("Data\\FFT\\Output_Resize_2.bin");
		
		tcomplex* d_output;
		cudaMalloc((void**)&d_output, (newdims.x / 2 + 1) * newdims.y * newdims.z * sizeof(tcomplex));
		tcomplex* d_intermediate;
		cudaMalloc((void**)&d_intermediate, (olddims.x / 2 + 1) * olddims.y * olddims.z * sizeof(tcomplex));

		d_FFTR2C(d_input, d_intermediate, DimensionCount(olddims), olddims);
		d_FFTPad(d_intermediate, d_output, olddims, newdims);

		tcomplex* h_output = (tcomplex*)MallocFromDeviceArray(d_output, (newdims.x / 2 + 1) * newdims.y * newdims.z * sizeof(tcomplex));
	
		double MeanRelative = GetMeanRelativeError((tfloat*)desired_output, (tfloat*)h_output, (newdims.x / 2 + 1) * newdims.y * newdims.z);
		ASSERT_LE(MeanRelative, 1e-5);

		cudaFree(d_input);
		//free(desired_output);
		free(h_output);
	}

	cudaDeviceReset();
}