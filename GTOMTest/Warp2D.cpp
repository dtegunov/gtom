#include "Prerequisites.h"

TEST(Transformation, Warp2D)
{
	cudaDeviceReset();

	//Case 1:
	{
		int2 dimsimage = toInt2(1024, 1024);
		int nframes = 40;
		tfloat* d_input = (tfloat*)CudaMallocFromBinaryFile("Data\\Transformation\\Input_Warp2D.bin");
		tfloat* d_output = CudaMallocValueFilled(Elements2(dimsimage) * nframes, (tfloat)0);

		int2 dimsgrid = toInt2(4, 4);
		tfloat2* h_grid = (tfloat2*)MallocValueFilled(Elements2(dimsgrid) * 2, (tfloat)0);

		for (uint n = 0; n < nframes; n++)
		{
			h_grid[1 * 4 + 0] = tfloat2(n * 1.0, 0.0);
			h_grid[1 * 4 + 1] = tfloat2(n * 1.0, 0.0);
			h_grid[1 * 4 + 2] = tfloat2(n * 1.0, 0.0);
			h_grid[1 * 4 + 3] = tfloat2(n * 1.0, 0.0);
			h_grid[2 * 4 + 0] = tfloat2(n * 1.0, 0.0);
			h_grid[2 * 4 + 1] = tfloat2(n * 1.0, 0.0);
			h_grid[2 * 4 + 2] = tfloat2(n * 1.0, 0.0);
			h_grid[2 * 4 + 3] = tfloat2(n * 1.0, 0.0);
			tfloat2* d_grid = (tfloat2*)CudaMallocFromHostArray(h_grid, Elements2(dimsgrid) * sizeof(tfloat2));

			d_Warp2D(d_input, dimsimage, d_grid, dimsgrid, d_output + Elements2(dimsimage) * n);

			cudaFree(d_grid);
		}

		d_WriteMRC(d_output, toInt3(dimsimage.x, dimsimage.y, nframes), "d_warped.mrc");
	}

	cudaDeviceReset();
}