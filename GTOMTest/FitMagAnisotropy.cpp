#include "Prerequisites.h"

TEST(Transformation, FitMagAnisotropy)
{
	cudaDeviceReset();

	//Case 1:
	{
		HeaderMRC refheader = ReadMRCHeader("E:\\carrie\\Particles\\micro\\sum.mrc");
		int3 dims = refheader.dimensions;
		uint nrefs = 1;
		void* h_refraw;
		ReadMRC("E:\\carrie\\Particles\\micro\\sum.mrc", &h_refraw);
		tfloat* h_ref = MixedToHostTfloat(h_refraw, refheader.mode, Elements(dims));

		float bestdistortion = 0;
		float bestangle = 0;
		d_FitMagAnisotropy(h_ref, toInt2(dims), 70, 0.03f, 0.0002f, ToRad(2.0f), bestdistortion, bestangle);

		bestangle = ToDeg(bestangle);
		std::cout << bestangle;
	}


	cudaDeviceReset();
}