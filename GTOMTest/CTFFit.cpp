#include "Prerequisites.h"

TEST(CTF, Fit)
{
	cudaDeviceReset();

	//Case 1:
	{
		int2 dimsimage = toInt2(2048, 2048);
		int2 dimsregion = toInt2(256, 256);

		CTFFitParams fp;
		fp.defocus = tfloat3(-2.0e-6f, 0.0e-6f, 0.1e-6f);
		fp.defocusdelta = tfloat3(0.0f, 1.0e-6f, 0.025e-6f);
		fp.dimsperiodogram = dimsregion;
		fp.maskinnerradius = 16;
		fp.maskouterradius = 64;
		CTFParams params;
		params.pixelsize = 1.35e-10;
		params.defocus = -0.5e-6;
		params.decayspread = 0;

		tfloat* d_input = (tfloat*)CudaMallocFromBinaryFile("Data/CTF/Input_Periodogram.bin");
		/*tfloat* h_input = 0;
		ReadMRC("Data/CTF/t0.mrc", (void**)&h_input, MRC_DATATYPE::MRC_FLOAT);
		tfloat* d_input = (tfloat*)CudaMallocFromHostArray(h_input, Elements2(dimsimage) * sizeof(tfloat));*/
		tfloat* d_output = CudaMallocValueFilled(ElementsFFT2(dimsregion), (tfloat)0);

		tfloat score = 0, mean = 0, stddev = 0;

		CTFParams bla;
		CTFParamsLean blalean = CTFParamsLean(bla);

		d_CTFFit(d_input, dimsimage, 0.5f, params, fp, 2, params, score, mean, stddev);
		cout << params.defocus;
	}

	cudaDeviceReset();
}