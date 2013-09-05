#include "Prerequisites.h"

TEST(FFT_1D, Zeros_512)
{
	int3 dimensions = {512, 1, 1};

	int reallength = dimensions.x;
	int complexlength = dimensions.x / 2 + 1;

	float* h_input = (float*)malloc(reallength * sizeof(float));
	for(int i = 0; i < reallength; i++)
		h_input[i] = 0.0f;

	float* d_input;
	cudaMalloc((void**)&d_input, reallength * sizeof(float));
	cudaMemcpy(d_input, h_input, reallength * sizeof(float), cudaMemcpyHostToDevice);

	float* d_output;
	cudaMalloc((void**)&d_output, complexlength * sizeof(cufftComplex));

	FFT((cufftReal*)d_input, (cufftComplex*)d_output, 1, dimensions);

	float* h_output = (float*)malloc(complexlength * sizeof(cufftComplex));
	cudaMemcpy(h_output, d_output, complexlength * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

	cudaFree(d_input);
	cudaFree(d_output);
	free(h_input);

	for(int i = 0; i < complexlength; i++)
		ASSERT_FLOAT_EQ(0.0f, h_output[i]);
			

	free(h_output);
}

TEST(FFT_1D, Zeros_1024)
{
	int3 dimensions = {1024, 1, 1};

	int reallength = dimensions.x;
	int complexlength = dimensions.x / 2 + 1;

	float* h_input = (float*)malloc(reallength * sizeof(float));
	for(int i = 0; i < reallength; i++)
		h_input[i] = 0.0f;

	float* d_input;
	cudaMalloc((void**)&d_input, reallength * sizeof(float));
	cudaMemcpy(d_input, h_input, reallength * sizeof(float), cudaMemcpyHostToDevice);

	float* d_output;
	cudaMalloc((void**)&d_output, complexlength * sizeof(cufftComplex));

	FFT((cufftReal*)d_input, (cufftComplex*)d_output, 1, dimensions);

	float* h_output = (float*)malloc(complexlength * sizeof(cufftComplex));
	cudaMemcpy(h_output, d_output, complexlength * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

	cudaFree(d_input);
	cudaFree(d_output);
	free(h_input);

	for(int i = 0; i < complexlength; i++)
		ASSERT_FLOAT_EQ(0.0f, h_output[i]);

	free(h_output);
}

TEST(FFT_1D, Zeros_2048)
{
	int3 dimensions = {2048, 1, 1};

	int reallength = dimensions.x;
	int complexlength = dimensions.x / 2 + 1;

	float* h_input = (float*)malloc(reallength * sizeof(float));
	for(int i = 0; i < reallength; i++)
		h_input[i] = 0.0f;

	float* d_input;
	cudaMalloc((void**)&d_input, reallength * sizeof(float));
	cudaMemcpy(d_input, h_input, reallength * sizeof(float), cudaMemcpyHostToDevice);

	float* d_output;
	cudaMalloc((void**)&d_output, complexlength * sizeof(cufftComplex));

	FFT((cufftReal*)d_input, (cufftComplex*)d_output, 1, dimensions);

	float* h_output = (float*)malloc(complexlength * sizeof(cufftComplex));
	cudaMemcpy(h_output, d_output, complexlength * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

	cudaFree(d_input);
	cudaFree(d_output);
	free(h_input);

	for(int i = 0; i < complexlength; i++)
		ASSERT_FLOAT_EQ(0.0f, h_output[i]);

	free(h_output);
}

TEST(FFT_1D, Data_512)
{
	int3 dimensions = {512, 1, 1};

	int reallength = dimensions.x;
	int complexlength = dimensions.x / 2 + 1;

	float* h_input = (float*)LoadArrayFromBinary("Data\\FFT\\Input_1D_512.bin");

	float* d_input;
	cudaMalloc((void**)&d_input, reallength * sizeof(float));
	cudaMemcpy(d_input, h_input, reallength * sizeof(float), cudaMemcpyHostToDevice);

	float* d_output;
	cudaMalloc((void**)&d_output, complexlength * sizeof(cufftComplex));

	FFT((cufftReal*)d_input, (cufftComplex*)d_output, 1, dimensions);

	float* h_output = (float*)malloc(complexlength * sizeof(cufftComplex));
	cudaMemcpy(h_output, d_output, complexlength * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

	float* desired_output = (float*)LoadArrayFromBinary("Data\\FFT\\Output_1D_512.bin");

	cudaFree(d_input);
	cudaFree(d_output);
	free(h_input);

	for(int i = 0; i < complexlength * 2; i++)
		ASSERT_RELATIVE_RANGE(desired_output[i], h_output[i], 1e-5f);
			
	printf("\tMean absolute error: %e\n", GetMeanAbsoluteError(desired_output, h_output, complexlength * 2));
	printf("\tMean relative error: %e\n\n", GetMeanRelativeError(desired_output, h_output, complexlength * 2));

	free(h_output);
	free(desired_output);
}

TEST(FFT_1D, Data_1024)
{
	int3 dimensions = {1024, 1, 1};

	int reallength = dimensions.x;
	int complexlength = dimensions.x / 2 + 1;

	float* h_input = (float*)LoadArrayFromBinary("Data\\FFT\\Input_1D_1024.bin");

	float* d_input;
	cudaMalloc((void**)&d_input, reallength * sizeof(float));
	cudaMemcpy(d_input, h_input, reallength * sizeof(float), cudaMemcpyHostToDevice);

	float* d_output;
	cudaMalloc((void**)&d_output, complexlength * sizeof(cufftComplex));

	FFT((cufftReal*)d_input, (cufftComplex*)d_output, 1, dimensions);

	float* h_output = (float*)malloc(complexlength * sizeof(cufftComplex));
	cudaMemcpy(h_output, d_output, complexlength * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

	float* desired_output = (float*)LoadArrayFromBinary("Data\\FFT\\Output_1D_1024.bin");

	cudaFree(d_input);
	cudaFree(d_output);
	free(h_input);

	for(int i = 0; i < complexlength * 2; i++)
		ASSERT_RELATIVE_RANGE(desired_output[i], h_output[i], 5e-5f);
			
	printf("\tMean absolute error: %e\n", GetMeanAbsoluteError(desired_output, h_output, complexlength * 2));
	printf("\tMean relative error: %e\n\n", GetMeanRelativeError(desired_output, h_output, complexlength * 2));

	free(h_output);
	free(desired_output);
}