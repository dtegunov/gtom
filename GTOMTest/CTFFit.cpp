#include "Prerequisites.h"

TEST(CTF, CTFFit)
{
	cudaDeviceReset();

	//Case 1:
	{
		int2 dimsimage = toInt2(3838, 3710);
		int2 dimsregion = toInt2(256, 256);

		CTFFitParams fp;
		fp.pixelsize = tfloat3(3.42e-10f);
		fp.defocus = tfloat3(-7.0e-6f, -5.0e-6f, 0.1e-6f);
		fp.defocusdelta = tfloat3(0.0f, 1.0e-6f, 0.025e-6f);
		fp.astigmatismangle = tfloat3(0.0f, PI, ToRad(20.0f));
		fp.dimsperiodogram = dimsregion;
		fp.maskinnerradius = 16;
		fp.maskouterradius = 64;
		CTFParams fit;

		//tfloat* d_input = (tfloat*)CudaMallocFromBinaryFile("Data/CTF/Input_Periodogram.bin");
		tfloat* h_input = 0;
		ReadMRC("Data/CTF/t0.mrc", (void**)&h_input, MRC_DATATYPE::MRC_FLOAT);
		tfloat* d_input = (tfloat*)CudaMallocFromHostArray(h_input, Elements2(dimsimage) * sizeof(tfloat));
		tfloat* d_output = CudaMallocValueFilled(ElementsFFT2(dimsregion), (tfloat)0);

		tfloat score = 0, mean = 0, stddev = 0;

		CTFParams bla;
		CTFParamsLean blalean = CTFParamsLean(bla);

		d_CTFFit(d_input, dimsimage, NULL, 0, fp, 2, fit, score, mean, stddev);
		cout << fit.defocus;
	}

	cudaDeviceReset();
}