#include "Prerequisites.h"

TEST(Generics, Dev)
{
	for(int i = 12; i < 13; i++)
	{	
		cudaDeviceReset();

		srand(i);
		int size = (1<<i);
		int batch = 30;

		tfloat* h_input = (tfloat*)malloc(size * size * batch * sizeof(tfloat));
		for(int b = 0; b < batch; b++)
		{
			for(int j = 0; j < size * size; j++)
				h_input[b * size * size + j] = j % 2 == 0 ? (tfloat)(b + 1) : (tfloat)0;
		}
		tfloat* d_input = (tfloat*)CudaMallocFromHostArray(h_input, size * size * batch * sizeof(tfloat));

		imgstats5* d_result;
		cudaMalloc((void**)&d_result, batch * sizeof(imgstats5));

		CUDA_MEASURE_TIME(d_Dev(d_input, d_result, size * size, (int*)NULL, batch));

		imgstats5* h_result = (imgstats5*)MallocFromDeviceArray(d_result, batch * sizeof(imgstats5));
		for(int b = 0; b < 1; b++)
			printf("Mean = %f, max = %f, min = %f, stddev = %f, var = %f\n", 
				   h_result[b].mean, 
				   h_result[b].min, 
				   h_result[b].max, 
				   h_result[b].stddev, 
				   h_result[b].var);

		cudaFree(d_input);
		cudaFree(d_result);
		free(h_input);
		free(h_result);

		cudaDeviceReset();
	}
}