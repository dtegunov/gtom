#include "Prerequisites.h"

TEST(CTF, Decay)
{
	cudaDeviceReset();

	//Case 1:
	{
		int2 dimsimage = toInt2(16, 16);

		tfloat* h_input = (tfloat*)malloc(Elements2(dimsimage) * sizeof(tfloat));
		for (int y = 0; y < dimsimage.y; y++)
			for (int x = 0; x < dimsimage.x; x++)
				h_input[y * dimsimage.x + x] = x * y;
		tfloat* d_input = (tfloat*)CudaMallocFromHostArray(h_input, Elements2(dimsimage) * sizeof(tfloat));
		tfloat* d_output = CudaMallocValueFilled(Elements2(dimsimage), (tfloat)0);

		d_CTFDecay(d_input, d_output, dimsimage, 4, 1);
		CudaWriteToBinaryFile("d_decay.bin", d_output, Elements2(dimsimage) * sizeof(tfloat));
	}

	cudaDeviceReset();
}