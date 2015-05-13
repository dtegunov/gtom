#include "Prerequisites.h"

TEST(CTF, Wiener)
{
	cudaDeviceReset();

	//Case 1:
	{
		int3 dimsinput = toInt3(280, 280, 1);
		tcomplex* d_inputctf = CudaMallocValueFilled(ElementsFFT(dimsinput), make_cuComplex(1, 1));;
		tfloat* d_inputfsc = CudaMallocValueFilled(dimsinput.x / 2, (tfloat)1);

		CTFParams params;
		params.pixelsize = 1.35e-10;
		params.decayspread = 0;
		params.defocus = -1e-6;
		params.amplitude = 0.1;

		tcomplex* d_corrected = CudaMallocValueFilled(ElementsFFT(dimsinput), make_cuComplex(3.14, 3.14));
		tfloat* d_correctedweights = CudaMallocValueFilled(ElementsFFT(dimsinput), (tfloat)2.72);

		d_CTFWiener(d_inputctf, dimsinput, d_inputfsc, &params, d_corrected, d_correctedweights);

		tfloat* h_corrected = (tfloat*)MallocFromDeviceArray(d_corrected, ElementsFFT(dimsinput) * sizeof(tfloat));
		free(h_corrected);

		tfloat* h_correctedweights = (tfloat*)MallocFromDeviceArray(d_correctedweights, ElementsFFT(dimsinput) * sizeof(tfloat));
		free(h_correctedweights);
	}

	cudaDeviceReset();
}