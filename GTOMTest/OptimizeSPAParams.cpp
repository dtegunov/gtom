#include "Prerequisites.h"

TEST(Optimization, OptimizeSPAParams)
{
	cudaDeviceReset();

	//Case 1:
	{
		int3 dimsimage = toInt3(64, 64, 1);
		int nimages = 5;

		tfloat* d_images = (tfloat*)CudaMallocFromBinaryFile("Data\\Optimization\\Input_SPAParams_images.bin");
		tfloat* d_imagespsf = (tfloat*)CudaMallocFromBinaryFile("Data\\Optimization\\Input_SPAParams_psf.bin");

		tfloat3* h_angles = (tfloat3*)MallocValueFilled(nimages * 3, (tfloat)0);
		tfloat2* h_shifts = (tfloat2*)MallocValueFilled(nimages * 2, (tfloat)0);

		srand(124);

		h_angles[0].y = ToRad(-2.0f);
		h_angles[1].y = ToRad(-1.0f);
		h_angles[2].y = ToRad(0.0f);
		h_angles[3].y = ToRad(-0.5f);
		h_angles[4].y = ToRad(-1.5f);

		for (int n = 0; n < nimages; n++)
		{
			h_angles[n].x = ToRad((float)rand() / (float)RAND_MAX - 0.5f) * 4.0f;
			h_angles[n].y = ToRad((float)rand() / (float)RAND_MAX - 0.5f) * 4.0f;
			h_angles[n].z = ToRad((float)rand() / (float)RAND_MAX - 0.5f) * 4.0f;
		}

		int maskradius = 24;
		tfloat2 thetarange = tfloat2(ToRad(-90.0f), ToRad(90.0f));
		tfloat finalscore;

		d_OptimizeSPAParams(d_images, d_imagespsf, toInt2(dimsimage), nimages, 0, maskradius, h_angles, h_shifts, NULL, tfloat3(0), 0, tfloat2(0, 0), finalscore);

		cout << finalscore;

		free(h_angles);
		free(h_shifts);
	}

	cudaDeviceReset();
}