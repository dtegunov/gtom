#include "Prerequisites.h"

template<class T> T KahanSum(T *data, int size)
{
    T sum = data[0];
    T c = (T)0.0;

    for (int i = 1; i < size; i++)
    {
        T y = data[i] - c;
        T t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }

    return sum;
}

TEST(Reductions, Sum)
{
	for(int i = 2; i < 28; i++)
	{
		int size = (1<<i) - 1;
		srand(i);

		tfloat* h_input = MallocValueFilled<tfloat>(size, 1);
		for(int j = 0; j < size; j++)
			h_input[j] = (tfloat)(rand() % 100000) * 0.00001f;

		tfloat preciseSum = KahanSum<tfloat>(h_input, size);
		tfloat naiveSum = 0;
		for(int j = 0; j < size; j++)
			naiveSum += h_input[j];

		tfloat* d_input = (tfloat*)CudaMallocFromHostArray(h_input, size * sizeof(tfloat));
		tfloat* d_output;
		cudaMalloc((void**)&d_output, sizeof(tfloat));

		CUDA_MEASURE_TIME(d_Sum(d_input, d_output, size, 1));

		tfloat gpuResult = 0;
		cudaMemcpy(&gpuResult, d_output, sizeof(tfloat), cudaMemcpyDeviceToHost);

		printf("Size = %d, precise result = %f, naive = %f, gpu = %f\n", size, preciseSum, naiveSum, gpuResult);
		printf("Error naive = %f, gpu = %f\n", abs(naiveSum - preciseSum) / preciseSum, abs(gpuResult - preciseSum) / preciseSum);
		ASSERT_LE(abs(gpuResult - preciseSum) / preciseSum, 0.000001);

		free(h_input);
		cudaFree(d_input);
		cudaFree(d_output);
	}
}