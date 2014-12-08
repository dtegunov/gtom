#include "Prerequisites.h"

TEST(Reductions, Min)
{
	for(int i = 10; i < 28; i++)
	{		
		int size = (1<<i) - (std::rand() % i);
		srand(i);

		tuple2<tfloat, size_t> cpuResult(9999, 0);
		tfloat* h_input = MallocValueFilled<tfloat>(size, 1);
		for(int j = 0; j < size; j++)
			h_input[j] = (tfloat)(std::rand() % 100000) * 0.00001f - 0.5f;
		h_input[std::rand() % size] = -0.500001f;
		for(int j = 0; j < size; j++)
			if(h_input[j] < cpuResult.t1)
				cpuResult = tuple2<tfloat, size_t>(h_input[j], j);

		tfloat* d_input = (tfloat*)CudaMallocFromHostArray(h_input, size * sizeof(tfloat));
		tuple2<tfloat, size_t>* d_output;
		cudaMalloc((void**)&d_output, sizeof(tuple2<tfloat, size_t>));

		CUDA_MEASURE_TIME(d_Min(d_input, d_output, size, 1));

		tuple2<tfloat, size_t> gpuResult = tuple2<tfloat, size_t>(0, 0);
		cudaMemcpy(&gpuResult, d_output, sizeof(tuple2<tfloat, size_t>), cudaMemcpyDeviceToHost);

		printf("Size = %d, cpu result = %f at %d, gpu result = %f at %d\n", size, cpuResult.t1, cpuResult.t2, gpuResult.t1, gpuResult.t2);
		ASSERT_EQ(cpuResult.t2, gpuResult.t2);

		free(h_input);
		cudaFree(d_input);
		cudaFree(d_output);
	}
}