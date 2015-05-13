#include "Prerequisites.h"

TEST(Optimization, OptimizeStackWarp2D)
{
	cudaDeviceReset();

	//Case 1:
	{
		HeaderMRC header = ReadMRCHeader("Data\\Optimization\\stack_27-Mar-2015_16-59-47_movie.running.mrc");
		int2 dimsimage = toInt2(header.dimensions);
		uint nimages = header.dimensions.z;
		//nimages = 10;
		int2 dimsgrid = toInt2(4, 4);

		void* h_mrcraw;
		ReadMRC("Data\\Optimization\\stack_27-Mar-2015_16-59-47_movie.running.mrc", &h_mrcraw);

		tfloat* d_images = MixedToDeviceTfloat(h_mrcraw, header.mode, Elements2(dimsimage) * nimages);

		tfloat2* h_grid = (tfloat2*)MallocValueFilled(Elements2(dimsgrid) * nimages * 2, (tfloat)0);
		tfloat* h_scores = MallocValueFilled(nimages, (tfloat)0);

		d_OptimizeStackWarp2D(d_images, dimsimage, dimsgrid, nimages, 100, 48, h_grid, h_scores);

		WriteToBinaryFile("d_grid.bin", h_grid, Elements2(dimsgrid) * nimages * 2 * sizeof(tfloat));
	}

	cudaDeviceReset();
}