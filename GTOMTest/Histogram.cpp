#include "Prerequisites.h"

TEST(Generics, Histogram)
{
	cudaDeviceReset();

	int range = 64;
	int copies = 65536;

	tfloat* h_input = (tfloat*)malloc(range * copies * sizeof(tfloat));
	for (int i = 0; i < range * copies; i++)
		h_input[i] = (tfloat)(i % range);

	tfloat* d_input = (tfloat*)CudaMallocFromHostArray(h_input, range * copies * sizeof(tfloat));
	uint* d_histogram = CudaMallocValueFilled(range, (uint)0);

	d_Histogram(d_input, d_histogram, range * copies, range, (tfloat)0, (tfloat)(range));

	uint* h_histogram = (uint*)MallocFromDeviceArray(d_histogram, range * sizeof(uint));
	free(h_histogram);

	cudaFree(d_histogram);
	cudaFree(d_input);

	cudaDeviceReset();
}