#include "Prerequisites.cuh"
#include "CTF.cuh"
#include "CubicInterp.cuh"
#include "FFT.cuh"
#include "Generics.cuh"
#include "Helper.cuh"
#include "Optimization.cuh"
#include "Transformation.cuh"


namespace gtom
{
	template<int maxbins> __global__ void CTFRotationalAverageKernel(tfloat* d_input, float2* d_inputcoords, tfloat* d_average, tfloat* d_averageweights, uint inputlength, uint sidelength, ushort numbins, ushort freqlow, ushort freqhigh, CTFParamsLean* d_params);


	////////////////////////////////////////////////////////////
	//Correct the CTF function to make all amplitudes positive//
	////////////////////////////////////////////////////////////

	void d_CTFRotationalAverage(tfloat* d_re, int2 dimsinput, CTFParams* h_params, tfloat* d_average, ushort freqlow, ushort freqhigh, int batch)
	{
		float2* h_targetcoords = (float2*)malloc(ElementsFFT2(dimsinput) * sizeof(float2));
		float invhalfsize = 2.0f / (float)dimsinput.x;
		float center = dimsinput.x / 2;
		for (int y = 0; y < dimsinput.y; y++)
		{
			for (int x = 0; x < ElementsFFT1(dimsinput.x); x++)
			{
				float2 point = make_float2(x - center, y - center);
				float angle = atan2(point.y, point.x);
				h_targetcoords[y * ElementsFFT1(dimsinput.x) + x] = make_float2(sqrt(point.x * point.x + point.y * point.y) * invhalfsize, angle);
			}
		}
		float2* d_targetcoords = (float2*)CudaMallocFromHostArray(h_targetcoords, ElementsFFT2(dimsinput) * sizeof(float2));
		free(h_targetcoords);

		d_CTFRotationalAverage(d_re, d_targetcoords, ElementsFFT2(dimsinput), dimsinput.x, h_params, d_average, freqlow, freqhigh, batch);
		cudaFree(d_targetcoords);
	}

	void d_CTFRotationalAverage(tfloat* d_input, float2* d_inputcoords, uint inputlength, uint sidelength, CTFParams* h_params, tfloat* d_average, ushort freqlow, ushort freqhigh, int batch)
	{
		uint numbins = freqhigh - freqlow;

		CTFParamsLean* h_lean = (CTFParamsLean*)malloc(batch * sizeof(CTFParamsLean));
		for (uint i = 0; i < batch; i++)
			h_lean[i] = CTFParamsLean(h_params[i]);
		CTFParamsLean* d_lean = (CTFParamsLean*)CudaMallocFromHostArray(h_lean, batch * sizeof(CTFParamsLean));

		dim3 TpB = dim3(192);
		dim3 grid = dim3(min(32, (inputlength + TpB.x - 1) / TpB.x), batch);

		tfloat* d_tempbins, *d_tempweights;
		cudaMalloc((void**)&d_tempbins, numbins * grid.x * grid.y * sizeof(tfloat));
		cudaMalloc((void**)&d_tempweights, numbins * grid.x * grid.y * sizeof(tfloat));

		if (numbins <= 513)
			CTFRotationalAverageKernel<513> << <grid, TpB >> > (d_input, d_inputcoords, d_tempbins, d_tempweights, inputlength, sidelength, numbins, freqlow, freqhigh, d_lean);
		else if (numbins <= 1025)
			CTFRotationalAverageKernel<1025> << <grid, TpB >> > (d_input, d_inputcoords, d_tempbins, d_tempweights, inputlength, sidelength, numbins, freqlow, freqhigh, d_lean);
		else if (numbins <= 2049)
			CTFRotationalAverageKernel<2049> << <grid, TpB >> > (d_input, d_inputcoords, d_tempbins, d_tempweights, inputlength, sidelength, numbins, freqlow, freqhigh, d_lean);
		else if (numbins <= 4097)
			CTFRotationalAverageKernel<4097> << <grid, TpB >> > (d_input, d_inputcoords, d_tempbins, d_tempweights, inputlength, sidelength, numbins, freqlow, freqhigh, d_lean);
		else
			throw;

		d_ReduceMeanWeighted(d_tempbins, d_tempweights, d_average, numbins, grid.x, batch);
		//cudaMemcpy(d_average, d_tempbins, numbins * batch * sizeof(tfloat), cudaMemcpyDeviceToDevice);

		cudaFree(d_tempweights);
		cudaFree(d_tempbins);
		cudaFree(d_lean);
		free(h_lean);
	}


	////////////////
	//CUDA kernels//
	////////////////

	template<int maxbins> __global__ void CTFRotationalAverageKernel(tfloat* d_input, float2* d_inputcoords, tfloat* d_average, tfloat* d_averageweights, uint inputlength, uint sidelength, ushort numbins, ushort freqlow, ushort freqhigh, CTFParamsLean* d_params)
	{
		__shared__ tfloat s_bins[maxbins], s_weights[maxbins];
		for (ushort i = threadIdx.x; i < numbins; i += blockDim.x)
		{
			s_bins[i] = 0;
			s_weights[i] = 0;
		}
		__syncthreads();

		CTFParamsLean p = d_params[blockIdx.y];
		d_input += blockIdx.y * inputlength;
		d_average += blockIdx.y * gridDim.x * numbins;
		d_averageweights += blockIdx.y * gridDim.x * numbins;

		double cs2 = p.Cs * p.Cs;
		double defocus2 = p.defocus * p.defocus;
		double lambda2 = p.lambda * p.lambda;
		double lambda4 = lambda2 * lambda2;

		for (uint id = blockIdx.x * blockDim.x + threadIdx.x; id < inputlength; id += gridDim.x * blockDim.x)
		{
			float radius = d_inputcoords[id].x;
			float angle = d_inputcoords[id].y;

			radius *= p.ny;
			double radius2 = radius * radius;
			double radius4 = radius2 * radius2;

			double term = p.defocus + p.defocusdelta * sin(2.0f * (angle - (float)p.astigmatismangle));
			term = 2.0 * p.Cs * lambda2 * radius2 * term;
			term = sqrt(defocus2 + cs2 * lambda4 * radius4 - term);
			term = sqrt(p.Cs * abs(abs(p.defocus) - term)) / (p.Cs * p.lambda);
			term /= p.ny * 2.0 / (double)sidelength;

			tfloat val = d_input[id];
			short lowbin = floor(term), highbin = lowbin + 1;
			tfloat lowweight = (tfloat)(1 + lowbin) - term, highweight = (tfloat)1 - lowweight;
			if (lowbin >= freqlow && lowbin < freqhigh)
			{
				lowbin -= freqlow;
				atomicAdd(s_bins + lowbin, val * lowweight);
				atomicAdd(s_weights + lowbin, lowweight);
			}
			if (highbin >= freqlow && highbin < freqhigh)
			{
				highbin -= freqlow;
				atomicAdd(s_bins + highbin, val * highweight);
				atomicAdd(s_weights + highbin, highweight);
			}
		}
		__syncthreads();

		d_average += blockIdx.x * numbins;
		d_averageweights += blockIdx.x * numbins;
		for (ushort i = threadIdx.x; i < numbins; i += blockDim.x)
		{
			d_average[i] = s_weights[i] != 0 ? s_bins[i] / s_weights[i] : 0;
			d_averageweights[i] = s_weights[i];
		}
	}
}