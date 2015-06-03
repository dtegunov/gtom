#include "Prerequisites.h"

TEST(CTF, Fit)
{
	cudaDeviceReset();

	//Case 1:
	{
		for (uint b = 0; b < 30; b++)
		{
			int2 dimsregion = toInt2(512, 512);

			CTFFitParams fp;
			fp.defocus = tfloat3(-0.5e-6f, 0.5e-6f, 0.05e-6f);
			fp.defocusdelta = tfloat3(0.0f, 0.5e-6f, 0.05e-6f);
			fp.phaseshift = tfloat3(0.0f, 1.0f * PI, 0.025f * PI);
			fp.astigmatismangle = tfloat3(-PI, PI, 0.2f * PI);
			fp.dimsperiodogram = dimsregion;
			fp.maskinnerradius = 24;
			fp.maskouterradius = 192;
			CTFParams params;
			params.pixelsize = 2.96e-10;
			params.defocus = -3.0e-6;
			params.voltage = 200e3f;
			params.Cs = 2.1e-3f;

			HeaderMRC header = ReadMRCHeader("D:/Dev/PP/PP_series4_def3um-01.mrc");
			int2 dimsimage = toInt2(header.dimensions);

			void* h_mrcraw = 0;
			ReadMRC("D:/Dev/PP/PP_series4_def3um-01.mrc", &h_mrcraw);
			tfloat* d_input = MixedToDeviceTfloat(h_mrcraw, header.mode, Elements2(dimsimage));
			cudaFreeHost(h_mrcraw);

			tfloat score = 0, mean = 0, stddev = 0;

			CTFParams bla;
			CTFParamsLean blalean = CTFParamsLean(bla);

			d_CTFFit(d_input, dimsimage, 0.5f, params, fp, 2, params, score, mean, stddev);
			cout << params.defocus;
			cudaFree(d_input);
		}
	}

	cudaDeviceReset();
}