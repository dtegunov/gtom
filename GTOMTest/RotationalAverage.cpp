#include "Prerequisites.h"

TEST(CTF, RotationalAverage)
{
	cudaDeviceReset();

	//Case 1:
	//{
	//	int2 dimsimage = toInt2(1024, 1024);
	//	
	//	tfloat* d_input = (tfloat*)CudaMallocFromBinaryFile("Data/CTF/Input_RadialAverage.bin");
	//	tfloat* d_output = CudaMallocValueFilled(ElementsFFT1(dimsimage.x), (tfloat)0);
	//	CTFParams params;
	//	params.defocus = -1.0e-6;
	//	params.defocusdelta = 0.5e-6;
	//	params.astigmatismangle = ToRad(45.0f);
	//	params.pixelsize = 1.35e-10;

	//	/*uint cutoff = CTFGetAliasingCutoff(params, 256);
	//	tfloat* h_sin = (tfloat*)malloc(128 * sizeof(tfloat));
	//	for (int i = 0; i < 128; i++)
	//		h_sin[i] = sin((float)i * PI2 / 9.0);
	//	tfloat* h_envelopemin = MallocValueFilled(128, 3.1415f);
	//	tfloat* h_envelopemax = MallocValueFilled(128, 3.1415f);
	//	h_CTFFitEnvelope(h_sin, 128, h_envelopemin, h_envelopemax, 2, 0, 127, 1);

	//	CTFParamsLean lean = CTFParamsLean(params);
	//	cout << lean.ny;*/

	//	d_CTFRotationalAverage(d_input, dimsimage, &params, d_output, 0, ElementsFFT1(dimsimage.x));

	//	tfloat* h_output = (tfloat*)MallocFromDeviceArray(d_output, ElementsFFT1(dimsimage.x) * sizeof(tfloat));
	//	free(h_output);
	//	CudaWriteToBinaryFile("d_rotaverage.bin", d_output, ElementsFFT1(dimsimage.x) * sizeof(tfloat));
	//}

	//Case 2:
	{
		int2 dims = toInt2(1024, 1024);
		int batch = 10;
		tfloat* d_input = (tfloat*)CudaMallocFromBinaryFile("Data/CTF/Input_RadialAverage2.bin");
		float2* h_coords = (float2*)malloc(Elements2(dims) * sizeof(float2));
		for (int y = 0; y < dims.y; y++)
		{
			int yy = y - dims.y / 2;
			for (int x = 0; x < dims.x; x++)
			{
				int xx = x - dims.x / 2;
				h_coords[y * dims.x + x] = make_float2(sqrt(xx * xx + yy * yy), atan2(yy, xx));
			}
		}
		float2* d_coords = (float2*)CudaMallocFromHostArray(h_coords, Elements2(dims) * sizeof(float2));

		CTFParams* h_source = (CTFParams*)malloc(batch * sizeof(CTFParams));
		for (int i = 0; i < 10; i++)
		{
			CTFParams source;
			source.defocus = -(1 + i * 0.01) * 1e-6;
			source.defocusdelta = 0.2e-6;
			source.astigmatismangle = ToRad(45.0f);
			source.pixeldelta = 0.1e-10;
			source.Cs = 2e-3;
			h_source[i] = source;
		}

		CTFParams target;
		target.defocus = -1e-6;
		target.Cs = 2e-3;

		tfloat* d_average = CudaMallocValueFilled(dims.x / 2, 0.0f);

		d_CTFRotationalAverageToTarget(d_input, d_coords, Elements2(dims), dims.x, h_source, target, d_average, 0, dims.x / 2, NULL, batch);

		d_WriteMRC(d_average, toInt3(dims.x / 2, 1, 1), "d_rotationalaverage.mrc");

		tfloat* h_average = (tfloat*)MallocFromDeviceArray(d_average, dims.x / 2 * sizeof(tfloat));
		cudaMemcpy(h_coords, d_coords, Elements2(dims) * sizeof(float2), cudaMemcpyDeviceToHost);
		free(h_average);

	}

	cudaDeviceReset();
}