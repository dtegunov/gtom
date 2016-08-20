#include "Prerequisites.cuh"
#include "FFT.cuh"
#include "Generics.cuh"
#include "Helper.cuh"
#include "ImageManipulation.cuh"
#include "Masking.cuh"


namespace gtom
{
	__global__ void BandpassNonCubicKernel(tcomplex* d_inputft, tcomplex* d_outputft, int3 dims, tfloat nyquistlow, tfloat nyquisthigh);

	///////////////////////////////////////////
	//Equivalent of TOM's tom_bandpass method//
	///////////////////////////////////////////

	void d_Bandpass(tfloat* d_input, tfloat* d_output, int3 dims, tfloat low, tfloat high, tfloat smooth, tfloat* d_mask, cufftHandle* planforw, cufftHandle* planback, int batch)
	{
		tcomplex* d_inputft;
		cudaMalloc((void**)&d_inputft, ElementsFFT(dims) * batch * sizeof(tcomplex));

		if (planforw == NULL)
			d_FFTR2C(d_input, d_inputft, DimensionCount(dims), dims, batch);
		else
			d_FFTR2C(d_input, d_inputft, planforw);

		d_Bandpass(d_inputft, d_inputft, dims, low, high, smooth, d_mask, batch);

		if (planback == NULL)
			d_IFFTC2R(d_inputft, d_output, DimensionCount(dims), dims, batch);
		else
			d_IFFTC2R(d_inputft, d_output, planback);

		cudaFree(d_inputft);
	}

	void d_Bandpass(tcomplex* d_inputft, tcomplex* d_outputft, int3 dims, tfloat low, tfloat high, tfloat smooth, tfloat* d_mask, int batch)
	{
		int dimensions = DimensionCount(dims);

		//Prepare mask:

		tfloat* d_localmask;

		if (d_mask == NULL)
		{
			tfloat* d_maskhigh = (tfloat*)CudaMallocValueFilled(Elements(dims), (tfloat)1);

			d_SphereMask(d_maskhigh, d_maskhigh, dims, &high, smooth, (tfloat3*)NULL, 1);

			tfloat* d_maskhighFFT;
			cudaMalloc((void**)&d_maskhighFFT, ElementsFFT(dims) * sizeof(tfloat));
			d_RemapFull2HalfFFT(d_maskhigh, d_maskhighFFT, dims);

			d_localmask = d_maskhighFFT;

			tfloat* d_masklowFFT;
			if (low > 0)
			{
				tfloat* d_masklow = (tfloat*)CudaMallocValueFilled(Elements(dims), (tfloat)1);
				d_SphereMask(d_masklow, d_masklow, dims, &low, smooth, (tfloat3*)NULL, 1);
				cudaMalloc((void**)&d_masklowFFT, ElementsFFT(dims) * sizeof(tfloat));
				d_RemapFull2HalfFFT(d_masklow, d_masklowFFT, dims);
				d_SubtractVector(d_localmask, d_masklowFFT, d_localmask, ElementsFFT(dims), 1);

				cudaFree(d_masklow);
				cudaFree(d_masklowFFT);
			}

			cudaFree(d_maskhigh);
		}
		else
			d_localmask = d_mask;

		//Mask FFT:

		d_ComplexMultiplyByVector(d_inputft, d_localmask, d_outputft, ElementsFFT(dims), batch);

		if (d_mask == NULL)
			cudaFree(d_localmask);
	}

	void d_BandpassNonCubic(tfloat* d_input, tfloat* d_output, int3 dims, tfloat nyquistlow, tfloat nyquisthigh, uint batch)
	{
		tcomplex* d_inputft;
		cudaMalloc((void**)&d_inputft, ElementsFFT(dims) * sizeof(tcomplex));
		d_FFTR2C(d_input, d_inputft, DimensionCount(dims), dims, batch);

		dim3 grid = dim3(dims.y, dims.z, batch);
		BandpassNonCubicKernel <<<grid, 128>>> (d_inputft, d_inputft, dims, nyquistlow, nyquisthigh);

		d_IFFTC2R(d_inputft, d_output, DimensionCount(dims), dims, batch);
		cudaFree(d_inputft);
	}

	__global__ void BandpassNonCubicKernel(tcomplex* d_inputft, tcomplex* d_outputft, int3 dims, tfloat nyquistlow, tfloat nyquisthigh)
	{
		int y = blockIdx.x;
		int z = blockIdx.y;

		d_inputft += ElementsFFT(dims) * blockIdx.z + (z * dims.y + y) * (dims.x / 2 + 1);
		d_outputft += ElementsFFT(dims) * blockIdx.z + (z * dims.y + y) * (dims.x / 2 + 1);

		float yy = y >= dims.y / 2 + 1 ? y - dims.y : y;
		yy /= dims.y / 2.0f;
		yy *= yy;
		float zz = z >= dims.z / 2 + 1 ? z - dims.z : z;
		zz /= dims.z / 2.0f;
		zz *= zz;

		for (int x = threadIdx.x; x < dims.x / 2 + 1; x += blockDim.x)
		{
			float xx = x;
			xx /= dims.x / 2.0f;
			xx *= xx;

			float r = sqrt(xx + yy + zz);
			if (r >= nyquistlow && r < nyquisthigh)
				d_outputft[x] = d_inputft[x];
			else
				d_outputft[x] = make_cuComplex(0, 0);
		}
	}
}