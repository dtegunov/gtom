#include "Prerequisites.h"

TEST(Transformation, Scale)
{
	cudaDeviceReset();

	//Case 1:
	{
		int3 olddims = {4, 4, 1};
		int3 newdims = {8, 8, 1};
		tfloat* d_input = (tfloat*)CudaMallocFromBinaryFile("Data\\Transformation\\Input_Scale_1.bin");
		tfloat* desired_output = (tfloat*)MallocFromBinaryFile("Data\\Transformation\\Output_Scale_1.bin");
		tfloat* d_output;
		cudaMalloc((void**)&d_output, newdims.x * newdims.y * newdims.z * sizeof(tfloat));

		d_Scale(d_input, d_output, olddims, newdims, T_INTERP_MODE::T_INTERP_FOURIER);
		tfloat* h_output = (tfloat*)MallocFromDeviceArray(d_output, newdims.x * newdims.y * newdims.z * sizeof(tfloat));
	
		double MeanRelative = GetMeanRelativeError((tfloat*)desired_output, (tfloat*)h_output, newdims.x * newdims.y * newdims.z);
		ASSERT_LE(MeanRelative, 1e-5);

		cudaFree(d_input);
		cudaFree(d_output);
		free(desired_output);
		free(h_output);
	}

	//Case 2:
	{
		int3 olddims = {7, 7, 1};
		int3 newdims = {17, 17, 1};
		tfloat* d_input = (tfloat*)CudaMallocFromBinaryFile("Data\\Transformation\\Input_Scale_2.bin");
		tfloat* desired_output = (tfloat*)MallocFromBinaryFile("Data\\Transformation\\Output_Scale_2.bin");
		tfloat* d_output;
		cudaMalloc((void**)&d_output, newdims.x * newdims.y * newdims.z * sizeof(tfloat));

		d_Scale(d_input, d_output, olddims, newdims, T_INTERP_MODE::T_INTERP_FOURIER);
		tfloat* h_output = (tfloat*)MallocFromDeviceArray(d_output, newdims.x * newdims.y * newdims.z * sizeof(tfloat));
	
		double MeanRelative = GetMeanRelativeError((tfloat*)desired_output, (tfloat*)h_output, newdims.x * newdims.y * newdims.z);
		ASSERT_LE(MeanRelative, 1e-5);

		cudaFree(d_input);
		cudaFree(d_output);
		free(desired_output);
		free(h_output);
	}

	//Case 3:
	{
		int3 olddims = {16, 16, 1};
		int3 newdims = {8, 8, 1};
		tfloat* d_input = (tfloat*)CudaMallocFromBinaryFile("Data\\Transformation\\Input_Scale_3.bin");
		tfloat* desired_output = (tfloat*)MallocFromBinaryFile("Data\\Transformation\\Output_Scale_3.bin");
		tfloat* d_output;
		cudaMalloc((void**)&d_output, newdims.x * newdims.y * newdims.z * sizeof(tfloat));

		d_Scale(d_input, d_output, olddims, newdims, T_INTERP_MODE::T_INTERP_FOURIER);
		tfloat* h_output = (tfloat*)MallocFromDeviceArray(d_output, newdims.x * newdims.y * newdims.z * sizeof(tfloat));
	
		double MeanRelative = GetMeanRelativeError((tfloat*)desired_output, (tfloat*)h_output, newdims.x * newdims.y * newdims.z);
		ASSERT_LE(MeanRelative, 1e-5);

		cudaFree(d_input);
		cudaFree(d_output);
		free(desired_output);
		free(h_output);
	}

	//Case 4:
	{
		int3 olddims = {15, 15, 1};
		int3 newdims = {7, 7, 1};
		tfloat* d_input = (tfloat*)CudaMallocFromBinaryFile("Data\\Transformation\\Input_Scale_4.bin");
		tfloat* desired_output = (tfloat*)MallocFromBinaryFile("Data\\Transformation\\Output_Scale_4.bin");
		tfloat* d_output;
		cudaMalloc((void**)&d_output, newdims.x * newdims.y * newdims.z * sizeof(tfloat));

		d_Scale(d_input, d_output, olddims, newdims, T_INTERP_MODE::T_INTERP_FOURIER);
		tfloat* h_output = (tfloat*)MallocFromDeviceArray(d_output, newdims.x * newdims.y * newdims.z * sizeof(tfloat));
	
		double MeanRelative = GetMeanRelativeError((tfloat*)desired_output, (tfloat*)h_output, newdims.x * newdims.y * newdims.z);
		ASSERT_LE(MeanRelative, 1e-5);

		cudaFree(d_input);
		cudaFree(d_output);
		free(desired_output);
		free(h_output);
	}

	//Case 5:
	{
		int3 olddims = {7, 7, 7};
		int3 newdims = {15, 15, 15};
		tfloat* d_input = (tfloat*)CudaMallocFromBinaryFile("Data\\Transformation\\Input_Scale_5.bin");
		tfloat* desired_output = (tfloat*)MallocFromBinaryFile("Data\\Transformation\\Output_Scale_5.bin");
		tfloat* d_output;
		cudaMalloc((void**)&d_output, newdims.x * newdims.y * newdims.z * sizeof(tfloat));

		d_Scale(d_input, d_output, olddims, newdims, T_INTERP_MODE::T_INTERP_FOURIER);
		tfloat* h_output = (tfloat*)MallocFromDeviceArray(d_output, newdims.x * newdims.y * newdims.z * sizeof(tfloat));
	
		double MeanRelative = GetMeanRelativeError((tfloat*)desired_output, (tfloat*)h_output, newdims.x * newdims.y * newdims.z);
		ASSERT_LE(MeanRelative, 1e-5);

		cudaFree(d_input);
		cudaFree(d_output);
		free(desired_output);
		free(h_output);
	}

	//Case 6:
	{
		int3 olddims = {7, 1, 1};
		int3 newdims = {15, 1, 1};
		tfloat* d_input = (tfloat*)CudaMallocFromBinaryFile("Data\\Transformation\\Input_Scale_6.bin");
		tfloat* desired_output = (tfloat*)MallocFromBinaryFile("Data\\Transformation\\Output_Scale_6.bin");
		tfloat* d_output;
		cudaMalloc((void**)&d_output, newdims.x * newdims.y * newdims.z * sizeof(tfloat));

		d_Scale(d_input, d_output, olddims, newdims, T_INTERP_MODE::T_INTERP_FOURIER);
		tfloat* h_output = (tfloat*)MallocFromDeviceArray(d_output, newdims.x * newdims.y * newdims.z * sizeof(tfloat));
	
		double MeanRelative = GetMeanRelativeError((tfloat*)desired_output, (tfloat*)h_output, newdims.x * newdims.y * newdims.z);
		ASSERT_LE(MeanRelative, 1e-5);

		cudaFree(d_input);
		cudaFree(d_output);
		free(desired_output);
		free(h_output);
	}


	cudaDeviceReset();
}