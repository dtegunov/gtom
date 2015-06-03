#include "Prerequisites.h"

TEST(CTF, RotationalAverage)
{
	cudaDeviceReset();

	//Case 1:
	{
		int2 dimsimage = toInt2(1024, 1024);
		
		tfloat* d_input = (tfloat*)CudaMallocFromBinaryFile("Data/CTF/Input_RadialAverage.bin");
		tfloat* d_output = CudaMallocValueFilled(ElementsFFT1(dimsimage.x), (tfloat)0);
		CTFParams params;
		params.defocus = -1.0e-6;
		params.defocusdelta = 0.5e-6;
		params.astigmatismangle = ToRad(45.0f);
		params.pixelsize = 1.35e-10;

		/*uint cutoff = CTFGetAliasingCutoff(params, 256);
		tfloat* h_sin = (tfloat*)malloc(128 * sizeof(tfloat));
		for (int i = 0; i < 128; i++)
			h_sin[i] = sin((float)i * PI2 / 9.0);
		tfloat* h_envelopemin = MallocValueFilled(128, 3.1415f);
		tfloat* h_envelopemax = MallocValueFilled(128, 3.1415f);
		h_CTFFitEnvelope(h_sin, 128, h_envelopemin, h_envelopemax, 2, 0, 127, 1);

		CTFParamsLean lean = CTFParamsLean(params);
		cout << lean.ny;*/

		d_CTFRotationalAverage(d_input, dimsimage, &params, d_output, 0, ElementsFFT1(dimsimage.x));

		tfloat* h_output = (tfloat*)MallocFromDeviceArray(d_output, ElementsFFT1(dimsimage.x) * sizeof(tfloat));
		free(h_output);
		CudaWriteToBinaryFile("d_rotaverage.bin", d_output, ElementsFFT1(dimsimage.x) * sizeof(tfloat));
	}

	cudaDeviceReset();
}