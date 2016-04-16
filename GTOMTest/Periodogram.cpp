#include "Prerequisites.h"

TEST(CTF, Periodogram)
{
	cudaDeviceReset();

	//Case 1:
	{
		int2 dimsimage = toInt2(3838, 3710);
		int2 dimsregion = toInt2(512, 512);

		tfloat* d_input = (tfloat*)CudaMallocFromBinaryFile("Data/CTF/Input_Periodogram.bin");
		tfloat* d_output = CudaMallocValueFilled(ElementsFFT2(dimsregion), (tfloat)0);
		CTFParams params;
		params.pixelsize = 1.35e-10;
		params.defocus = -1.0e-6;
		params.astigmatismangle = ToRad(67.5);
		params.defocusdelta = 2.5e-7f;

		d_CTFPeriodogram(d_input, dimsimage, 0.0f, dimsregion, dimsregion, d_output);
		d_WriteMRC(d_output, toInt3(ElementsFFT1(dimsregion.x), dimsregion.y, 1), "d_periodogram.mrc");

		int2 dimsgrid;
		int3* h_origins = GetEqualGridSpacing(dimsimage, dimsregion, 0.5f, dimsgrid);
		int norigins = Elements2(dimsgrid);
		int3* d_origins = (int3*)CudaMallocFromHostArray(h_origins, norigins * sizeof(int3));
		CTFParams* h_params = (CTFParams*)malloc(norigins * sizeof(CTFParams));
		for (int i = 0; i < norigins; i++)
			h_params[i] = params;

		CTFFitParams fp;
		fp.maskinnerradius = 16;
		fp.maskouterradius = 64;
		fp.dimsperiodogram = dimsregion;
		fp.defocus = tfloat3(-1.5e-6f, 2.5e-6f, 0.1e-6f);
		//fp.defocusdelta = tfloat3(0.0f, 0.5e-6f, 0.025e-6f);

		int2 dimspolar = GetCart2PolarFFTSize(dimsregion);
		dimspolar.x = fp.maskouterradius - fp.maskinnerradius;

		tfloat* d_ps2dpolar = CudaMallocValueFilled(Elements2(dimspolar) * norigins, (tfloat)0);
		float2* d_ps2dcoords = (float2*)CudaMallocValueFilled(Elements2(dimspolar) * 2, 0.0f);
		d_CTFFitCreateTarget2D(d_input, dimsimage, d_origins, h_params, norigins, fp, d_ps2dpolar, d_ps2dcoords);

		CudaWriteToBinaryFile("d_ps2dcoords.bin", d_ps2dcoords, Elements2(dimspolar) * sizeof(float2));

		tfloat* d_ps1d = CudaMallocValueFilled(dimspolar.x * norigins, (tfloat)0);
		float2* d_ps1dcoords = (float2*)CudaMallocValueFilled(dimspolar.x * norigins * 2, 0.0f);
		d_CTFFitCreateTarget1D(d_ps2dpolar, d_ps2dcoords, dimspolar, h_params, norigins, fp, d_ps1d, d_ps1dcoords);

		vector<pair<tfloat, CTFParams>> fits;
		float score, mean, stddev;
		//d_CTFFit(d_ps2dpolar, d_ps2dcoords, dimspolar, h_params, norigins, fp, 1, fits, score, mean, stddev);
		d_CTFFit(d_ps1d, d_ps1dcoords, toInt2(dimspolar.x, 1), h_params, norigins, fp, 1, fits, score, mean, stddev);
	}

	cudaDeviceReset();
}