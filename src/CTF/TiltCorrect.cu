#include "Prerequisites.cuh"
#include "CTF.cuh"
#include "FFT.cuh"
#include "Generics.cuh"
#include "Helper.cuh"
#include "ImageManipulation.cuh"
#include "Optimization.cuh"
#include "Transformation.cuh"


namespace gtom
{
	__global__ void UpdateWithFilteredKernel(tfloat* d_filtered, float* d_defoci, tfloat* d_output, uint elements, float lowbin, float highbin, float binsize, uint nbins);

	//void d_CTFTiltCorrect(tfloat* d_image, int2 dimsimage, CTFTiltParams tiltparams, tfloat snr, tfloat* d_output)
	//{
	//	int2 dimsregion = toInt2(256, 256);
	//
	//	size_t memlimit = 128 << 20;
	//	uint batchsize = min((uint)Elements2(dimsimage), (uint)(memlimit / (ElementsFFT2(dimsregion) * sizeof(tcomplex))));
	//
	//	tfloat* d_extracted;
	//	cudaMalloc((void**)&d_extracted, Elements2(dimsregion) * batchsize * sizeof(tfloat));
	//	tcomplex* d_extractedft;
	//	cudaMalloc((void**)&d_extractedft, ElementsFFT2(dimsregion) * batchsize * sizeof(tcomplex));
	//
	//	cufftHandle planforw, planback;
	//	planforw = d_FFTR2CGetPlan(2, toInt3(dimsregion), batchsize);
	//	planback = d_IFFTC2RGetPlan(2, toInt3(dimsregion), batchsize);
	//
	//	int2 dimsgrid;
	//	int3* h_origins = GetEqualGridSpacing(dimsimage, toInt2(1, 1), 0.0f, dimsgrid);
	//	int3* h_originsforw = (int3*)MallocValueFilled(Elements2(dimsgrid) * 3, 0);
	//	int3* h_originsback = (int3*)MallocValueFilled(Elements2(dimsgrid) * 3, 0);
	//	for (uint i = 0; i < Elements2(dimsgrid); i++)
	//	{
	//		h_originsforw[i].x = max(0, min(h_origins[i].x - dimsregion.x / 2, dimsimage.x - dimsregion.x - 0));
	//		h_originsforw[i].y = max(0, min(h_origins[i].y - dimsregion.y / 2, dimsimage.y - dimsregion.y - 0));
	//
	//		h_originsback[i].x = dimsregion.x / 2 + (h_origins[i].x - dimsregion.x / 2) - h_originsforw[i].x;
	//		h_originsback[i].y = dimsregion.y / 2 + (h_origins[i].y - dimsregion.y / 2) - h_originsforw[i].y;
	//	}
	//	int3* d_originsforw = (int3*)CudaMallocFromHostArray(h_originsforw, Elements2(dimsgrid) * sizeof(int3));
	//	int3* d_originsback = (int3*)CudaMallocFromHostArray(h_originsback, Elements2(dimsgrid) * sizeof(int3));
	//	CTFParams* h_params = (CTFParams*)malloc(Elements2(dimsimage) * sizeof(CTFParams));
	//	tiltparams.GetParamsGrid2D(dimsimage, toInt2(1, 1), h_origins, Elements2(dimsgrid), h_params);
	//
	//	for (uint b = 0; b < Elements2(dimsimage); b += batchsize)
	//	{
	//		uint curbatch = min(batchsize, Elements2(dimsimage) - b);
	//
	//		d_ExtractMany(d_image, d_extracted, toInt3(dimsimage), toInt3(dimsregion), d_originsforw + b, curbatch);
	//		//d_NormMonolithic(d_extracted, d_extracted, Elements2(dimsregion), T_NORM_MEAN01STD, curbatch);
	//		//d_WriteMRC(d_extracted, toInt3(dimsregion.x, dimsregion.y, curbatch), "d_extracted.mrc");
	//		d_FFTR2C(d_extracted, d_extractedft, &planforw);
	//		//d_Abs(d_extractedft, d_extracted, ElementsFFT2(dimsregion) * curbatch);
	//		//d_WriteMRC(d_extracted, toInt3(dimsregion.x / 2 + 1, dimsregion.y, curbatch), "d_extractedft.mrc");
	//
	//		d_CTFWiener(d_extractedft, toInt3(dimsregion), snr, h_params + b, d_extractedft, NULL, curbatch);
	//		//d_Re(d_extractedft, d_extracted, ElementsFFT2(dimsregion) * curbatch);
	//		//d_WriteMRC(d_extracted, toInt3(dimsregion.x / 2 + 1, dimsregion.y, curbatch), "d_extractedftwiener.mrc");
	//
	//		d_IFFTC2R(d_extractedft, d_extracted, &planback);
	//		//d_WriteMRC(d_extracted, toInt3(dimsregion.x, dimsregion.y, curbatch), "d_extractedwiener.mrc");
	//		d_Extract(d_extracted, d_output + b, toInt3(dimsregion), toInt3(1, 1, 1), d_originsback + b, curbatch);
	//	}
	//
	//	free(h_params);
	//	free(h_originsback);
	//	free(h_originsforw);
	//	free(h_origins);
	//
	//	cufftDestroy(planforw);
	//	cufftDestroy(planback);
	//
	//	cudaFree(d_originsback);
	//	cudaFree(d_originsforw);
	//	cudaFree(d_extractedft);
	//	cudaFree(d_extracted);
	//}

	void d_CTFTiltCorrect(tfloat* d_image, int2 dimsimage, CTFTiltParams tiltparams, tfloat snr, tfloat* d_output)
	{
		// Determine batch size
		size_t memlimit = 384 << 20;
		uint batchsize = min((uint)Elements2(dimsimage), (uint)(memlimit / (ElementsFFT2(dimsimage) * sizeof(tcomplex))));
		if (batchsize < 2)
			throw;

		// Define grid of defocus values
		int2 dimsgrid;
		int3* h_origins = GetEqualGridSpacing(dimsimage, toInt2(1, 1), 0.0f, dimsgrid);
		float* h_defoci = (float*)malloc(Elements2(dimsgrid) * sizeof(float));
		tiltparams.GetZGrid2D(dimsimage, toInt2(1, 1), h_origins, Elements2(dimsgrid), h_defoci);
		float* d_defoci = (float*)CudaMallocFromHostArray(h_defoci, Elements2(dimsgrid) * sizeof(float));
		//d_WriteMRC(d_defoci, toInt3(dimsgrid), "d_defoci.mrc");

		// Determine min and max defocus in image, define defocus bins
		float zmin = 1e30f, zmax = -1e30f;
		for (uint i = 0; i < Elements2(dimsgrid); i++)
		{
			zmin = min(zmin, h_defoci[i]);
			zmax = max(zmax, h_defoci[i]);
		}
		float binsize = (zmax - zmin) / (float)max(dimsgrid.x, dimsgrid.y);
		binsize = max(binsize, tiltparams.centerparams.pixelsize * 4.0f);
		uint nbins = (uint)((zmax - zmin) / binsize) + 1;

		tcomplex* d_imageft;
		cudaMalloc((void**)&d_imageft, ElementsFFT2(dimsimage) * batchsize * sizeof(tcomplex));
		tcomplex* d_imageftfiltered;
		cudaMalloc((void**)&d_imageftfiltered, ElementsFFT2(dimsimage) * batchsize * sizeof(tcomplex));
		tfloat* d_imagefiltered;
		cudaMalloc((void**)&d_imagefiltered, Elements2(dimsimage) * batchsize * sizeof(tfloat));

		// FFT original image and make [batchsize] copies for filtering later
		d_FFTR2C(d_image, d_imageft, 2, toInt3(dimsimage));
		CudaMemcpyMulti(d_imageft + ElementsFFT2(dimsimage), d_imageft, ElementsFFT2(dimsimage), batchsize - 1);

		// Plan for IFFT of filtered spectra
		cufftHandle planback = d_IFFTC2RGetPlan(2, toInt3(dimsimage), batchsize);

		for (uint b = 0; b < nbins; b += batchsize - 1)
		{
			uint curbatch = min(nbins - b, batchsize);

			float lowbin = (float)b * binsize + zmin, highbin = (float)(b + curbatch - 1) * binsize + zmin;
			CTFParams* h_params = (CTFParams*)malloc(curbatch * sizeof(CTFParams));
			for (uint i = 0; i < curbatch; i++)
			{
				h_params[i] = tiltparams.centerparams;
				h_params[i].defocus = lowbin + (float)i * binsize;
			}

			d_CTFWiener(d_imageft, toInt3(dimsimage), snr, h_params, d_imageftfiltered, NULL, curbatch);
			d_IFFTC2R(d_imageftfiltered, d_imagefiltered, &planback);
			free(h_params);

			uint TpB = 256;
			dim3 grid = dim3(min(8192, (Elements2(dimsimage) + TpB - 1) / TpB), 1, 1);
			UpdateWithFilteredKernel << <grid, TpB >> > (d_imagefiltered, d_defoci, d_output, Elements2(dimsimage), lowbin, highbin, binsize, curbatch);
		}

		d_MultiplyByScalar(d_output, d_output, Elements2(dimsimage), (tfloat)1 / (tfloat)Elements2(dimsimage));

		// Clean up
		cufftDestroy(planback);

		cudaFree(d_imagefiltered);
		cudaFree(d_imageftfiltered);
		cudaFree(d_imageft);
		cudaFree(d_defoci);

		free(h_defoci);
		free(h_origins);
	}

	__global__ void UpdateWithFilteredKernel(tfloat* d_filtered, float* d_defoci, tfloat* d_output, uint elements, float lowbin, float highbin, float binsize, uint nbins)
	{
		for (uint id = blockIdx.x * blockDim.x + threadIdx.x; id < elements; id += gridDim.x * blockDim.x)
		{
			float defocus = d_defoci[id];
			if (defocus < lowbin || defocus > highbin)
				continue;

			defocus -= lowbin;
			uint bin = defocus / binsize;
			float interp = (defocus - (float)bin * binsize) / binsize;

			tfloat value = lerp(d_filtered[elements * bin + id], d_filtered[elements * min(nbins - 1, bin + 1) + id], interp);
			d_output[id] = value;
		}
	}
}