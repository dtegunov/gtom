#include "Prerequisites.h"

TEST(Correlation, Peak)
{
	cudaDeviceReset();

	//Case 1:
	{
		int3 dims = {8, 8, 1};
		tfloat* d_input = (tfloat*)CudaMallocFromBinaryFile("Data\\Correlation\\Input_Peak_1.bin");
		tfloat3* desired_output = (tfloat3*)MallocFromBinaryFile("Data\\Correlation\\Output_Peak_1.bin");
		tfloat3* d_positions;
		cudaMalloc((void**)&d_positions, sizeof(tfloat3));
		tfloat* d_values;
		cudaMalloc((void**)&d_values, sizeof(tfloat));

		d_Peak(d_input, d_positions, d_values, dims, T_PEAK_MODE::T_PEAK_INTEGER);

		tfloat3* h_positions = (tfloat3*)MallocFromDeviceArray(d_positions, sizeof(tfloat3));
	
		double MeanRelative = GetMeanRelativeError((tfloat*)desired_output, (tfloat*)h_positions, 3);
		ASSERT_LE(MeanRelative, 1e-5);

		cudaFree(d_input);
		cudaFree(d_positions);
		cudaFree(d_values);
		free(desired_output);
		free(h_positions);
	}

	//Case 2:
	{
		int3 dims = {20, 20, 1};
		tfloat* d_input = (tfloat*)CudaMallocFromBinaryFile("Data\\Correlation\\Input_Peak_2.bin");
		tfloat3* desired_output = (tfloat3*)MallocFromBinaryFile("Data\\Correlation\\Output_Peak_2.bin");
		tfloat3* d_positions;
		cudaMalloc((void**)&d_positions, sizeof(tfloat3));
		tfloat* d_values;
		cudaMalloc((void**)&d_values, sizeof(tfloat));

		d_Peak(d_input, d_positions, d_values, dims, T_PEAK_MODE::T_PEAK_SUBCOARSE);

		tfloat3* h_positions = (tfloat3*)MallocFromDeviceArray(d_positions, sizeof(tfloat3));
	
		double MeanRelative = GetMeanAbsoluteError((tfloat*)desired_output, (tfloat*)h_positions, DimensionCount(dims));
		ASSERT_LE(MeanRelative, 0.05);

		cudaFree(d_input);
		cudaFree(d_positions);
		cudaFree(d_values);
		free(desired_output);
		free(h_positions);
	}

	//Case 3:
	{
		int3 dims = {20, 20, 1};
		tfloat* d_input = (tfloat*)CudaMallocFromBinaryFile("Data\\Correlation\\Input_Peak_2.bin");
		tfloat3* desired_output = (tfloat3*)MallocFromBinaryFile("Data\\Correlation\\Output_Peak_2.bin");
		tfloat3* d_positions;
		cudaMalloc((void**)&d_positions, sizeof(tfloat3));
		tfloat* d_values;
		cudaMalloc((void**)&d_values, sizeof(tfloat));

		d_Peak(d_input, d_positions, d_values, dims, T_PEAK_MODE::T_PEAK_SUBFINE);

		tfloat3* h_positions = (tfloat3*)MallocFromDeviceArray(d_positions, sizeof(tfloat3));
	
		double MeanRelative = GetMeanAbsoluteError((tfloat*)desired_output, (tfloat*)h_positions, DimensionCount(dims));
		ASSERT_LE(MeanRelative, 0.02);

		cudaFree(d_input);
		cudaFree(d_positions);
		cudaFree(d_values);
		free(desired_output);
		free(h_positions);
	}

	cudaDeviceReset();
}