#include "Prerequisites.h"

TEST(CTF, CTFTiltFit)
{
	cudaDeviceReset();

	//Case 1:
	{
		int2 dimsgrid = toInt2(3838, 3710);
		tfloat2 spacing = tfloat2(3.42f, 3.42f);
		CTFTiltParams params = CTFTiltParams(tfloat3(ToRad(45.0f), ToRad(30.0f), 0.0f), CTFParams());
		params.centerparams.defocus = 0.0f;
		tfloat* zgrid = params.GetZGrid2D(dimsgrid, spacing, tfloat3(0.0f));
		WriteToBinaryFile("d_zgrid.bin", zgrid, Elements2(dimsgrid) * sizeof(tfloat));

		tfloat* h_input = 0;
		ReadMRC("Data/CTF/t0.mrc", (void**)&h_input, MRC_DATATYPE::MRC_FLOAT);
		tfloat* d_input = (tfloat*)CudaMallocFromHostArray(h_input, Elements2(dimsgrid) * sizeof(tfloat));
		CTFFitParams fp;
		fp.pixelsize = tfloat3(6.84e-10f);
		fp.defocus = tfloat3(-6.0e-6f, 6.0e-6f, 0.1e-6f);
		fp.defocusdelta = tfloat3(0.0f, 0.0f, 0.2e-6f);
		fp.astigmatismangle = tfloat3(0.0f, 0.0f, ToRad(45.0f));
		fp.dimsperiodogram = toInt2(256, 256);
		fp.maskinnerradius = 4;
		fp.maskouterradius = 64;

		tfloat score = 0, stddev = 0;
		d_CTFTiltFit(d_input, dimsgrid, fp, 2, 128, params, score, stddev);
	}

	cudaDeviceReset();
}