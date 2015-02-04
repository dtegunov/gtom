#include "Prerequisites.h"

TEST(Transformation, FFTLines)
{
	cudaDeviceReset();

	//Case 1:
	{
		int2 dimsimage = toInt2(32, 32);
		tfloat* d_input = (tfloat*)CudaMallocFromBinaryFile("Data\\Correlation\\Input_SimMatrix.bin");

		tcomplex* d_inputft;
		cudaMalloc((void**)&d_inputft, ElementsFFT2(dimsimage) * sizeof(tcomplex));
		d_FFTR2C(d_input, d_inputft, 2, toInt3(dimsimage), 1);

		int anglesteps = 36;
		int linewidth = 1;
		int2 dimslines = toInt2(dimsimage.x / 2 + 1, anglesteps * linewidth);

		tcomplex* d_lines;
		cudaMalloc((void**)&d_lines, Elements2(dimslines) * sizeof(tcomplex));

		d_FFTLines(d_inputft, d_lines, dimsimage, T_INTERP_CUBIC, anglesteps, linewidth, 1);

		CudaWriteToBinaryFile("d_lines.bin", d_lines, Elements2(dimslines) * sizeof(tcomplex));
	}

	cudaDeviceReset();
}