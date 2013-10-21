#include "Prerequisites.h"

TEST(FFT, FFTResize)
{
	cudaDeviceReset();

	//Case 1:
	{
		int3 olddims = {16, 17, 1};
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
		int3 olddims = {16, 17, 1};
		int3 newdims = {7, 8, 1};

		tfloat* d_input = (tfloat*)CudaMallocFromBinaryFile("Data\\FFT\\Input_Resize_2.bin");
		tfloat* desired_output = (tfloat*)MallocFromBinaryFile("Data\\FFT\\Output_Resize_2.bin");
		
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

	//Case 3:
	{
		int3 olddims = {16, 16, 1};
		int3 newdims = {7, 7, 1};

		tfloat* d_input = (tfloat*)CudaMallocFromBinaryFile("Data\\FFT\\Input_Resize_3.bin");
		tfloat* desired_output = (tfloat*)MallocFromBinaryFile("Data\\FFT\\Output_Resize_3.bin");
		
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

	//Case 4:
	{
		int3 olddims = {16, 16, 1};
		int3 newdims = {7, 8, 1};

		tfloat* d_input = (tfloat*)CudaMallocFromBinaryFile("Data\\FFT\\Input_Resize_4.bin");
		tfloat* desired_output = (tfloat*)MallocFromBinaryFile("Data\\FFT\\Output_Resize_4.bin");
		
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

	//Case 5:
	{
		int3 olddims = {8, 8, 1};
		int3 newdims = {16, 16, 1};

		tfloat* d_input = (tfloat*)CudaMallocFromBinaryFile("Data\\FFT\\Input_Resize_5.bin");
		tfloat* desired_output = (tfloat*)MallocFromBinaryFile("Data\\FFT\\Output_Resize_5.bin");
		
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

	//Case 6:
	{
		int3 olddims = {7, 8, 1};
		int3 newdims = {16, 16, 1};

		tfloat* d_input = (tfloat*)CudaMallocFromBinaryFile("Data\\FFT\\Input_Resize_6.bin");
		tfloat* desired_output = (tfloat*)MallocFromBinaryFile("Data\\FFT\\Output_Resize_6.bin");
		
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

	//Case 7:
	{
		int3 olddims = {8, 8, 1};
		int3 newdims = {16, 17, 1};

		tfloat* d_input = (tfloat*)CudaMallocFromBinaryFile("Data\\FFT\\Input_Resize_7.bin");
		tfloat* desired_output = (tfloat*)MallocFromBinaryFile("Data\\FFT\\Output_Resize_7.bin");
		
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

	//Case 8:
	{
		int3 olddims = {16, 16, 1};
		int3 newdims = {7, 7, 1};

		tfloat* d_input = (tfloat*)CudaMallocFromBinaryFile("Data\\FFT\\Input_Resize_8.bin");
		tfloat* desired_output = (tfloat*)MallocFromBinaryFile("Data\\FFT\\Output_Resize_8.bin");
		
		tcomplex* d_output;
		cudaMalloc((void**)&d_output, Elements(newdims) * sizeof(tcomplex));
		tcomplex* d_intermediate;
		cudaMalloc((void**)&d_intermediate, (olddims.x / 2 + 1) * olddims.y * olddims.z * sizeof(tcomplex));
		tcomplex* d_intermediate2;
		cudaMalloc((void**)&d_intermediate2, Elements(olddims) * sizeof(tcomplex));

		d_FFTR2C(d_input, d_intermediate, DimensionCount(olddims), olddims);
		d_HermitianSymmetryPad(d_intermediate, d_intermediate2, olddims);
		d_FFTFullCrop(d_intermediate2, d_output, olddims, newdims);

		tcomplex* h_output = (tcomplex*)MallocFromDeviceArray(d_output, Elements(newdims) * sizeof(tcomplex));
	
		double MeanRelative = GetMeanRelativeError((tfloat*)desired_output, (tfloat*)h_output, Elements(newdims) * 2);
		ASSERT_LE(MeanRelative, 1e-5);

		cudaFree(d_input);
		//free(desired_output);
		free(h_output);
	}

	cudaDeviceReset();
}