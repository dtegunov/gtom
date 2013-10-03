#include "Prerequisites.h"

TEST(ImageManipulation, Bin)
{
	for(int i = 11; i < 12; i++)
	{	
		cudaDeviceReset();

		srand(i);
		int size = (1<<i);
		int batch = 1;
		int bincount = 5;

		tfloat* h_input = (tfloat*)malloc(size * size * batch * sizeof(tfloat));
		for(int b = 0; b < batch; b++)
		{
			for(int j = 0; j < size * size; j++)
				h_input[b * size * size + j] = (tfloat)(j % (1<<bincount));
		}
		tfloat* d_input = (tfloat*)CudaMallocFromHostArray(h_input, size * size * batch * sizeof(tfloat));

		tfloat* d_result;
		cudaMalloc((void**)&d_result, size * size / (1<<(bincount * 2)) * sizeof(tfloat));

		int3 dims;
		dims.x = size;
		dims.y = size;
		d_Bin(d_input, d_result, dims, bincount, 1);

		tfloat* h_result = (tfloat*)MallocFromDeviceArray(d_result, size * size / (1<<(bincount * 2)) * sizeof(tfloat));

		ASSERT_ARRAY_EQ(h_result, (tfloat)((1<<bincount) - 1) / (tfloat)2, size * size / (1<<(bincount * 2)));

		cudaFree(d_input);
		cudaFree(d_result);
		free(h_input);
		free(h_result);

		cudaDeviceReset();
	}

	for(int i = 9; i < 10; i++)
	{	
		cudaDeviceReset();

		srand(i);
		size_t size = (1<<i);
		size_t batch = 1;
		size_t bincount = 2;

		tfloat* h_input;
		cudaMallocHost((void**)&h_input, size * size * size * batch * sizeof(tfloat), 0);
		for(int b = 0; b < batch; b++)
		{
			for(int j = 0; j < size * size * size; j++)
				h_input[b * size * size * size + j] = (tfloat)(j % (1<<bincount));
		}
		tfloat* d_input = (tfloat*)CudaMallocFromHostArray(h_input, size * size * size * batch * sizeof(tfloat));

		tfloat* d_result;
		cudaMalloc((void**)&d_result, size * size * size / (1<<(bincount * 3)) * batch * sizeof(tfloat));

		int3 dims;
		dims.x = size;
		dims.y = size;
		dims.z = size;
		d_Bin(d_input, d_result, dims, bincount, batch);

		tfloat* h_result = (tfloat*)MallocFromDeviceArray(d_result, size * size * size / (1<<(bincount * 3)) * batch * sizeof(tfloat));

		ASSERT_ARRAY_EQ(h_result, (tfloat)((1<<bincount) - 1) / (tfloat)2, size * size / (1<<(bincount * 2)));

		cudaFreeHost(h_input);
		free(h_result);
		cudaFree(d_input);
		cudaFree(d_result);

		cudaDeviceReset();
	}
}