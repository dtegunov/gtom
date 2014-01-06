#include "Prerequisites.h"

TEST(ImageManipulation, Norm)
{
	/*for(int i = 14; i < 15; i++)
	{	
		cudaDeviceReset();

		srand(i);
		int size = (1<<i);
		int batch = 1;

		tfloat* h_input = (tfloat*)malloc(size * size * batch * sizeof(tfloat));
		for(int b = 0; b < batch; b++)
		{
			for(int j = 0; j < size * size; j++)
				h_input[b * size * size + j] = j % 2 == 0 ? (tfloat)(b + 1) : (tfloat)0;
		}
		tfloat* d_input = (tfloat*)CudaMallocFromHostArray(h_input, size * size * batch * sizeof(tfloat));

		imgstats5* d_imagestats;
		cudaMalloc((void**)&d_imagestats, batch * sizeof(imgstats5));

		d_Dev(d_input, d_imagestats, size * size, (int*)NULL, batch);

		printf("Stats before normalization:\n");
		imgstats5* h_imagestats = (imgstats5*)MallocFromDeviceArray(d_imagestats, batch * sizeof(imgstats5));
		for(int b = 0; b < 1; b++)
			printf("Mean = %f, max = %f, min = %f, stddev = %f, var = %f\n", 
				   h_imagestats[b].mean, 
				   h_imagestats[b].min, 
				   h_imagestats[b].max, 
				   h_imagestats[b].stddev, 
				   h_imagestats[b].var);

		CUDA_MEASURE_TIME(d_Norm(d_input, d_input, size * size, (int*)NULL, T_NORM_MODE::T_NORM_CUSTOM, (tfloat)2, batch));

		d_Dev(d_input, d_imagestats, size * size, (int*)NULL, batch);

		printf("Stats after normalization:\n");
		h_imagestats = (imgstats5*)MallocFromDeviceArray(d_imagestats, batch * sizeof(imgstats5));
		for(int b = 0; b < 1; b++)
			printf("Mean = %f, max = %f, min = %f, stddev = %f, var = %f\n", 
				   h_imagestats[b].mean, 
				   h_imagestats[b].min, 
				   h_imagestats[b].max, 
				   h_imagestats[b].stddev, 
				   h_imagestats[b].var);

		cudaFree(d_input);
		cudaFree(d_imagestats);
		free(h_input);
		free(h_imagestats);

		cudaDeviceReset();
	}*/

	cudaDeviceReset();

	//Case 1:
	/*{
		int3 dims = {4, 3, 5};
		tfloat* d_input = (tfloat*)CudaMallocFromBinaryFile("Data\\ImageManipulation\\Input_Norm_1.bin");
		tfloat* desired_output = (tfloat*)MallocFromBinaryFile("Data\\ImageManipulation\\Output_Norm_1.bin");
		d_Norm(d_input, d_input, dims.x*dims.y*dims.z, (tfloat*)NULL, T_NORM_MODE::T_NORM_MEAN01STD, 0);
		tfloat* h_output = (tfloat*)MallocFromDeviceArray(d_input, dims.x * dims.y * dims.z * sizeof(tfloat));
	
		double MeanRelative = GetMeanRelativeError((tfloat*)desired_output, (tfloat*)h_output, dims.x * dims.y * dims.z);
		ASSERT_LE(MeanRelative, 1e-4);

		cudaFree(d_input);
		free(desired_output);
		free(h_output);
	}*/

	//Case 2:
	{
		int3 dims = {128, 128, 1};
		int batch = 4000;
		tfloat* d_input = (tfloat*)CudaMallocFromBinaryFile("Data\\ImageManipulation\\Input_Norm_2.bin");
		tfloat* desired_output = (tfloat*)MallocFromBinaryFile("Data\\ImageManipulation\\Output_Norm_2.bin");
		d_NormMonolithic(d_input, d_input, Elements(dims), T_NORM_MODE::T_NORM_MEAN01STD, batch);
		tfloat* h_output = (tfloat*)MallocFromDeviceArray(d_input, Elements(dims) * batch * sizeof(tfloat));
	
		double MeanRelative = GetMeanRelativeError((tfloat*)desired_output, (tfloat*)h_output, Elements(dims) * batch);
		ASSERT_LE(MeanRelative, 1e-3);

		cudaFree(d_input);
		free(desired_output);
		free(h_output);
	}

	cudaDeviceReset();
}