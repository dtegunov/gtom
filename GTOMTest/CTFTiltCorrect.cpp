#include "Prerequisites.h"

TEST(CTF, TiltCorrect)
{
	cudaDeviceReset();

	//Case 1:
	{
		/*HeaderMRC header = ReadMRCHeader("Data/CTF/stack_02-Nov-2013_21-50-06.dat.2-1.mrc");
		int2 dimsimage = toInt2(header.dimensions.x, header.dimensions.y);
		int2 dimsregion = toInt2(512, 512);
		void* h_mrcraw;
		ReadMRC("Data/CTF/stack_02-Nov-2013_21-50-06.dat.2-1.mrc", (void**)&h_mrcraw);
		tfloat* d_image = MixedToDeviceTfloat(h_mrcraw, header.mode, Elements(header.dimensions));
		cudaFree(h_mrcraw);
		tfloat* d_output = CudaMallocValueFilled(Elements2(dimsimage), (tfloat)0);

		CTFParams params;
		params.pixelsize = 1.35e-10;
		params.defocus = -1.0125e-6;
		params.astigmatismangle = 1.17035544;
		params.defocusdelta = 0.246875032e-10;
		params.decayspread = 0;

		CTFTiltParams tp(0.0f, tfloat2(0.0f, 0.0f), tfloat2(0.0f, 0.0f), params);

		d_CTFTiltCorrect(d_image, dimsimage, tp, 1.0f, d_output);

		d_WriteMRC(d_output, toInt3(dimsimage), "d_tiltcorrect.mrc");*/
	}

	cudaDeviceReset();

	//Case 2:
	{
		HeaderMRC header = ReadMRCHeader("Data/CTF/L3Tomo3_plus0.mrc");
		int2 dimsimage = toInt2(header.dimensions.x, header.dimensions.y);
		int2 dimsregion = toInt2(512, 512);
		void* h_mrcraw;
		ReadMRC("Data/CTF/L3Tomo3_plus0.mrc", (void**)&h_mrcraw);
		tfloat* d_image = MixedToDeviceTfloat(h_mrcraw, header.mode, Elements(header.dimensions));
		cudaFreeHost(h_mrcraw);
		tfloat* d_output = CudaMallocValueFilled(Elements2(dimsimage), (tfloat)0);

		CTFParams params;
		params.pixelsize = 3.42e-10;
		params.defocus = -5.6382164e-06;
		params.astigmatismangle = 0;
		params.defocusdelta = 0;
		params.decayspread = 0;

		CTFTiltParams tp(0.0f, tfloat2(-1.6915f, -7.8530e-05f), tfloat2(3.9924f, 0.1124f), params);

		d_CTFTiltCorrect(d_image, dimsimage, tp, 1.0f, d_output);

		d_WriteMRC(d_output, toInt3(dimsimage), "d_tiltcorrect.mrc");

		cudaFree(d_output);
		cudaFree(d_image);
	}

	cudaDeviceReset();
}