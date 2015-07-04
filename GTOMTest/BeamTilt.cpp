#include "Prerequisites.h"

TEST(ImageManipulation, BeamTilt)
{
	cudaDeviceReset();

	//Case 1:
	{
		int2 dims = toInt2(16, 16);

		tcomplex* h_input = MallocValueFilled(ElementsFFT2(dims), make_cuComplex(0, 0));
		for (uint i = 0; i < ElementsFFT2(dims); i++)
			h_input[i] = make_cuComplex(i % 20, i % 13);
		tcomplex* d_input = (tcomplex*)CudaMallocFromHostArray(h_input, ElementsFFT2(dims) * sizeof(tcomplex));

		CTFParams params;
		params.pixelsize = 1.35e-10;
		params.Cs = 2e-3;
		CTFParamsLean lean = CTFParamsLean(params);
		
		tfloat2 beamtilt = tfloat2(1, -2);
		tfloat2* d_beamtilt = (tfloat2*)CudaMallocFromHostArray(&beamtilt, sizeof(tfloat2));

		double boxsize = (params.pixelsize * 1e10) * (double)dims.x;
		double factor = 0.360 * lean.Cs * lean.lambda * lean.lambda / (boxsize * boxsize * boxsize);
		for (long int i = 0, ip = 0; i<dims.y; i++, ip = (i < dims.x / 2 + 1) ? i : i - dims.y) \
			for (long int j = 0, jp = 0; j<dims.x / 2 + 1; j++, jp = j)
		{
			double delta_phase = factor * (ip * ip + jp * jp) * (ip * beamtilt.y + jp * beamtilt.x);
			double realval = h_input[i * ElementsFFT1(dims.x) + j].x;
			double imagval = h_input[i * ElementsFFT1(dims.x) + j].y;
			double mag = sqrt(realval * realval + imagval * imagval);
			double phas = atan2(imagval, realval) + ToRad(delta_phase); // apply phase shift!
			realval = mag * cos(phas);
			imagval = mag * sin(phas);
			h_input[i * ElementsFFT1(dims.x) + j] = make_cuComplex(realval, imagval);
		}

		d_BeamTilt(d_input, d_input, dims, d_beamtilt, &params, 1);

		tcomplex* h_result = (tcomplex*)MallocFromDeviceArray(d_input, ElementsFFT2(dims) * sizeof(tcomplex));

		double diffsum = 0;
		for (uint i = 0; i < ElementsFFT2(dims); i++)
		{
			diffsum += abs(h_result[i].x - h_input[i].x);
			diffsum += abs(h_result[i].y - h_input[i].y);
		}

		assert(diffsum < 1e-4);
	}

	cudaDeviceReset();
}