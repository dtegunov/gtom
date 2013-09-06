#include "Prerequisites.h"

void FFT_Test_Zeros(int const ndimensions, int3 const dimensions)
{
	int reallength = dimensions.x * dimensions.y * dimensions.z;
	int complexlength = (dimensions.x / 2 + 1) * dimensions.y * dimensions.z;

	float* d_input = CudaMallocZeroFilledFloat(reallength);

	float* d_output;
	cudaMalloc((void**)&d_output, complexlength * sizeof(cufftComplex));

	float executiontime = 0;
	CUDA_MEASURE_TIME(FFT((cufftReal*)d_input, (cufftComplex*)d_output, ndimensions, dimensions), executiontime);

	float* h_output = (float*)MallocFromDeviceArray(d_output, complexlength * sizeof(cufftComplex));

	cudaFree(d_input);
	cudaFree(d_output);

	ASSERT_ARRAY_EQ(h_output, 0.0f, complexlength * 2);

	printf("Kernel executed in %f ms.\n", executiontime);

	free(h_output);
}

void FFT_Test_Data(string inpath, string outpath, int const ndimensions, int3 const dimensions)
{
	int reallength = dimensions.x * dimensions.y * dimensions.z;
	int complexlength = dimensions.x / 2 + 1 * dimensions.y * dimensions.z;

	float* d_input = (float*)CudaMallocFromBinaryFile(inpath);

	float* d_output;
	cudaMalloc((void**)&d_output, complexlength * sizeof(cufftComplex));

	float executiontime = 0;
	CUDA_MEASURE_TIME(FFT((cufftReal*)d_input, (cufftComplex*)d_output, ndimensions, dimensions), executiontime);

	float* h_output = (float*)MallocFromDeviceArray(d_output, complexlength * sizeof(cufftComplex));
	float* desired_output = (float*)MallocFromBinary(outpath);

	cudaFree(d_input);
	cudaFree(d_output);

	for(int i = 0; i < complexlength * 2; i++)
		ASSERT_RELATIVE_RANGE(desired_output[i], h_output[i], 5e-5f);
			
	printf("Mean absolute error: %e\n", GetMeanAbsoluteError(desired_output, h_output, complexlength * 2));
	printf("Mean relative error: %e\n", GetMeanRelativeError(desired_output, h_output, complexlength * 2));
	printf("Kernel executed in %f ms.\n", executiontime);

	free(h_output);
	free(desired_output);
}

TEST(FFT_1D, Zeros_512)
{
	int3 dimensions = {512, 1, 1};
	FFT_Test_Zeros(1, dimensions);
}

TEST(FFT_1D, Zeros_1024)
{
	int3 dimensions = {1024, 1, 1};
	FFT_Test_Zeros(1, dimensions);
}

TEST(FFT_1D, Zeros_2048)
{
	int3 dimensions = {2048, 1, 1};
	FFT_Test_Zeros(1, dimensions);
}

TEST(FFT_1D, Data_512)
{
	int3 dimensions = {512, 1, 1};
	FFT_Test_Data("Data\\FFT\\Input_1D_512.bin", "Data\\FFT\\Output_1D_512.bin", 1, dimensions);
}

TEST(FFT_1D, Data_1024)
{
	int3 dimensions = {1024, 1, 1};
	FFT_Test_Data("Data\\FFT\\Input_1D_1024.bin", "Data\\FFT\\Output_1D_1024.bin", 1, dimensions);
}

TEST(FFT_2D, Zeros_512)
{
	int3 dimensions = {512, 512, 1};
	FFT_Test_Zeros(2, dimensions);
}

TEST(FFT_2D, Zeros_1024)
{
	int3 dimensions = {1024, 1024, 1};
	FFT_Test_Zeros(2, dimensions);
}

TEST(FFT_2D, Zeros_2048)
{
	int3 dimensions = {2048, 2048, 1};
	FFT_Test_Zeros(2, dimensions);
}

TEST(FFT_2D, Zeros_4096)
{
	int3 dimensions = {4096, 4096, 1};
	FFT_Test_Zeros(2, dimensions);
}

TEST(FFT_2D, Zeros_8192)
{
	int3 dimensions = {8192, 8192, 1};
	FFT_Test_Zeros(2, dimensions);
}

TEST(FFT_2D, Zeros_16384)
{
	int3 dimensions = {16384, 16384, 1};
	FFT_Test_Zeros(2, dimensions);
}