#include "Prerequisites.h"

TEST(CTF, CTFCorrect)
{
	cudaDeviceReset();

	//Case 1:
	{
		int2 dimsimage = toInt2(2048, 2048);
		int2 dimsregion = toInt2(512, 512);

		CTFFitParams fp;
		fp.pixelsize = tfloat3(1.35e-10f);
		fp.defocus = tfloat3(-4.0e-6f, 0.0f, 0.1e-6f);
		fp.defocusdelta = tfloat3(0.0f, 0.6e-6f, 0.05e-6f);
		fp.astigmatismangle = tfloat3(0.0f, PI, ToRad(10.0f));
		fp.dimsperiodogram = dimsregion;
		CTFParams fit;

		tfloat* d_input = (tfloat*)CudaMallocFromBinaryFile("Data/CTF/Input_Periodogram.bin");
		tfloat* d_output = CudaMallocValueFilled(ElementsFFT2(dimsregion), (tfloat)0);

		tfloat score = 0, mean = 0, stddev = 0;

		d_CTFFit(d_input, dimsimage, NULL, 0, fp, 2, fit, score, mean, stddev);

		tcomplex* d_inputft;
		cudaMalloc((void**)&d_inputft, ElementsFFT2(dimsimage) * sizeof(tcomplex));
		d_FFTR2C(d_input, d_inputft, 2, toInt3(dimsimage));

		d_CTFCorrect(d_inputft, toInt3(dimsimage), fit, d_inputft);
		d_IFFTC2R(d_inputft, d_input, 2, toInt3(dimsimage));
		/*d_CTFCorrect(d_input, toInt3(dimsimage), fit, d_input);*/

		CudaWriteToBinaryFile("d_ctfcorrected.bin", d_input, Elements2(dimsimage) * sizeof(tfloat));
	}

	cudaDeviceReset();
}