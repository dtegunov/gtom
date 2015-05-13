#include "Prerequisites.cuh"
#include "Angles.cuh"
#include "Correlation.cuh"
#include "CTF.cuh"
#include "FFT.cuh"
#include "Generics.cuh"
#include "Helper.cuh"
#include "ImageManipulation.cuh"
#include "Masking.cuh"
#include "Optimization.cuh"
#include "Transformation.cuh"


////////////////////////////
//CUDA kernel declarations//
////////////////////////////

__global__ void AccumulateSpectraKernel(tfloat* d_ps1d, tfloat* d_defoci, uint nspectra, uint length, tfloat* d_accumulated, tfloat accumulateddefocus, tfloat* d_perbatchoffsets, uint lowfreq, uint relevantlength, double cs, double lambda, double pxfactor);


/////////////////////////////////
//Auxiliary methods and kernels//
/////////////////////////////////

void PopulateAngles(vector<tfloat3> &v_angles, tfloat3 phibracket, tfloat3 thetabracket, tfloat3 psibracket)
{
	for (tfloat psi = psibracket.x; psi <= psibracket.y + 1e-5f; psi += psibracket.z)
	{
		for (tfloat theta = thetabracket.x; theta <= thetabracket.y + 1e-5f; theta += thetabracket.z)
		{
			for (tfloat phi = phibracket.x; phi <= phibracket.y + 1e-5f; phi += phibracket.z)
			{
				v_angles.push_back(tfloat3(phi, theta, psi));
				if (phibracket.z == 0)
					break;
			}
			if (thetabracket.z == 0)
				break;
		}
		if (psibracket.z == 0)
			break;
	}
}

void d_AccumulateSpectra(tfloat* d_ps1d, tfloat* d_defoci, uint nspectra, tfloat* d_accumulated, tfloat accumulateddefocus, tfloat* d_perbatchoffsets, CTFParams p, CTFFitParams fp, uint batch)
{
	uint length = fp.dimsperiodogram.x / 2;
	uint relevantlength = fp.maskouterradius - fp.maskinnerradius;
	CTFParamsLean lean = CTFParamsLean(p);
	double pxfactor = lean.ny / (double)length;

	dim3 TpB = dim3(min(128, NextMultipleOf(relevantlength, 32)));
	dim3 grid = dim3((relevantlength + TpB.x - 1) / TpB.x, batch);

	AccumulateSpectraKernel << <grid, TpB >> > (d_ps1d, d_defoci, nspectra, length, d_accumulated, accumulateddefocus, d_perbatchoffsets, fp.maskinnerradius, relevantlength, lean.Cs, lean.lambda, pxfactor);
}

__global__ void AccumulateSpectraKernel(tfloat* d_ps1d, tfloat* d_defoci, uint nspectra, uint length, tfloat* d_accumulated, tfloat accumulateddefocus, tfloat* d_perbatchoffsets, uint lowfreq, uint relevantlength, double cs, double lambda, double pxfactor)
{
	uint id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= relevantlength)
		return;
	
	tfloat sum = 0;
	uint samples = 0;
	double K = (double)(id + lowfreq) * pxfactor;
	double K2 = K * K;
	double K4 = K2 * K2;
	double defocusoffset = d_perbatchoffsets[blockIdx.y];
	double D = (accumulateddefocus + defocusoffset) * 1e10;
	double lambda2 = lambda * lambda;
	double lambda4 = lambda2 * lambda2;
	double cs2 = cs * cs;

	for (uint n = 0; n < nspectra; n++, d_ps1d += length)
	{
		double d = (d_defoci[n] + defocusoffset) * 1e10;
		if (d < 0 != D < 0)		// Combining defoci with different signs won't work
			continue;
		double k = sqrt(abs(abs(d) - sqrt(cs2 * K4 * lambda4 - 2.0 * cs * D * K2 * lambda2 + d * d)) / (cs * lambda2));
		k /= pxfactor;
		if (ceil(k) >= length)	// Out of range
			continue;

		// Cubic interpolation
		uint p1 = k;
		tfloat sample0 = d_ps1d[max(1U, p1) - 1];
		tfloat sample1 = d_ps1d[p1];
		tfloat sample2 = d_ps1d[min(length - 1, p1 + 1)];
		tfloat sample3 = d_ps1d[min(length - 1, p1 + 2)];

		tfloat factor0 = -0.5f * sample0 + 1.5f * sample1 - 1.5f * sample2 + 0.5f * sample3;
		tfloat factor1 = sample0 - 2.5f * sample1 + 2.0f * sample2 - 0.5f * sample3;
		tfloat factor2 = -0.5f * sample0 + 0.5f * sample2;
		tfloat factor3 = sample1;

		tfloat interp = k - (tfloat)p1;

		sum += ((factor0 * interp + factor1) * interp + factor2) * interp + factor3;
		samples++;
	}

	d_accumulated[relevantlength * blockIdx.y + id] = sum / (tfloat)max(1U, samples);
}

///////////////////////////////////////////
//Fit defocus and tilt in tilted specimen//
///////////////////////////////////////////

void h_CTFTiltFit(tfloat* h_image, int2 dimsimage, uint nimages, float overlapfraction, vector<CTFTiltParams> &startparams, CTFFitParams fp, tfloat maxtheta, tfloat2 &specimentilt, tfloat* h_defoci)
{
	tfloat* d_image;
	cudaMalloc((void**)&d_image, Elements2(dimsimage) * sizeof(tfloat));
	/*tfloat* d_imagecropped;
	cudaMalloc((void**)&d_imagecropped, Elements2(dimsimage) * sizeof(tfloat));*/

	tfloat3 phibracket = tfloat3(0.0f, ToRad(360.0f), ToRad(20.0f));
	tfloat3 thetabracket = tfloat3(0.0f, maxtheta, ToRad(4.0f));
	for (uint i = 0; i < nimages; i++)
		h_defoci[i] = 0;

	for (uint r = 0; r < 4; r++)
	{
		int2 anglegrid = toInt2((phibracket.y - phibracket.x + 1e-5f) / phibracket.z + 1, (thetabracket.y - thetabracket.x + 1e-5f) / thetabracket.z + 1);
		vector<tfloat3> v_angles;
		PopulateAngles(v_angles, phibracket, thetabracket, tfloat3(0));
		tfloat* h_scores = MallocValueFilled(Elements2(anglegrid), (tfloat)0);
		tfloat* h_samples = MallocValueFilled(Elements2(anglegrid), (tfloat)0);
		if (Elements2(anglegrid) != v_angles.size())
			throw;

		for (uint i = 0; i < nimages; i++)
		{
			cudaMemcpy(d_image, h_image + Elements2(dimsimage) * i, Elements2(dimsimage) * sizeof(tfloat), cudaMemcpyHostToDevice);
			/*int2 dimscropped = toInt2(cos(startparams[i].stageangle.y) * dimsimage.x, dimsimage.y);
			dimscropped.x += dimscropped.x % 2;
			d_Pad(d_image, d_imagecropped, toInt3(dimsimage), toInt3(dimscropped), T_PAD_VALUE, (tfloat)0);*/

			vector<tfloat2> v_results;
			CTFTiltParams adjustedparams(startparams[i].imageangle, startparams[i].stageangle, startparams[i].specimenangle, startparams[i].centerparams);
			adjustedparams.centerparams.defocus += h_defoci[i];
			d_CTFTiltFit(d_image, dimsimage, overlapfraction, adjustedparams, fp, v_angles, 3, v_results);

			// First, add to the average score grid...
			for (uint n = 0; n < Elements2(anglegrid); n++)
			{
				h_samples[n] += 1.0;// pow(cos(startparams[i].stageangle.y), 2.0);
				h_scores[n] += v_results[n].x;// *pow(cos(startparams[i].stageangle.y), 2.0);
			}
			// ... then, take the defocus value with the highest score
			sort(v_results.begin(), v_results.end(),
				[](const tfloat2 &a, const tfloat2 &b) -> bool
			{
				return a.x > b.x;
			});
			h_defoci[i] += v_results[0].y;	// Update adjustment for next iteration so it doesn't start the search from scratch
		}
		for (uint n = 0; n < Elements2(anglegrid); n++)
			h_scores[n] /= h_samples[n];
		//WriteToBinaryFile("d_scores.bin", h_scores, Elements2(anglegrid) * sizeof(tfloat));
		free(h_samples);

		int2 maxposition;
		tfloat maxvalue = -1e30f;
		for (uint y = 0; y < anglegrid.y; y++)
			for (uint x = 0; x < anglegrid.x; x++)
				if (h_scores[y * anglegrid.x + x] > maxvalue)
				{
					maxposition = toInt2(x, y);
					maxvalue = h_scores[y * anglegrid.x + x];
				}
		free(h_scores);

		// Update tilt estimate and shrink search brackets
		specimentilt = tfloat2(phibracket.x + maxposition.x * phibracket.z, thetabracket.x + maxposition.y * thetabracket.z);
		phibracket = tfloat3(specimentilt.x - phibracket.z * 0.75f, specimentilt.x + phibracket.z * 0.75f, phibracket.z / 4.0f);
		thetabracket = tfloat3(specimentilt.y - thetabracket.z * 0.75f, specimentilt.y + thetabracket.z * 0.75f, thetabracket.z / 4.0f);

		// Adjust defocus search bracket after first iteration, as the first defocus estimate can't be too far from the truth
		if (r == 0)
		{
			fp.defocus.x = -fp.defocus.z * 1.5;
			fp.defocus.y = fp.defocus.z * 1.5;
			fp.defocus.z /= 4.0;
		}
	}

	//cudaFree(d_imagecropped);
	cudaFree(d_image);
}

void d_CTFTiltFit(tfloat* d_image, int2 dimsimage, float overlapfraction, CTFTiltParams &startparams, CTFFitParams fp, vector<tfloat3> &v_angles, int defocusrefinements, vector<tfloat2> &v_results)
{
	CTFFitParams originalfp = fp;
	fp.maskinnerradius = 0;
	fp.maskouterradius = fp.dimsperiodogram.x / 2;
	int2 dimspolar = GetCart2PolarFFTSize(fp.dimsperiodogram);
	dimspolar.x = fp.maskouterradius - fp.maskinnerradius;

	// Create grid, allocate memory for spectra
	int2 dimsgrid;
	int3* h_origins = GetEqualGridSpacing(dimsimage, fp.dimsperiodogram, overlapfraction, dimsgrid);
	uint norigins = Elements2(dimsgrid);

	// Allocate memory for spectra and point coords
	tfloat* d_ps1d;
	cudaMalloc((void**)&d_ps1d, dimspolar.x * norigins * sizeof(tfloat));
	float2* d_ps1dcoords;
	cudaMalloc((void**)&d_ps1dcoords, dimspolar.x * norigins * sizeof(float2));

	{
		int3* d_origins = (int3*)CudaMallocFromHostArray(h_origins, norigins * sizeof(int3));
		tfloat* d_ps1dmin;
		cudaMalloc((void**)&d_ps1dmin, dimspolar.x * sizeof(tfloat));
		tfloat* d_ps1dmax;
		cudaMalloc((void**)&d_ps1dmax, dimspolar.x * sizeof(tfloat));
		tfloat* d_ps2d;		// Extracted spectra in Cartesian coords
		cudaMalloc((void**)&d_ps2d, ElementsFFT2(fp.dimsperiodogram) * norigins * sizeof(tfloat));
		tfloat* d_ps2dpolar;
		cudaMalloc((void**)&d_ps2dpolar, Elements2(dimspolar) * norigins * sizeof(tfloat));
		float2* d_ps2dcoords;
		cudaMalloc((void**)&d_ps2dcoords, Elements2(dimspolar) * sizeof(float2));

		CTFParams* h_params = (CTFParams*)malloc(norigins * sizeof(CTFParams));
		for (uint n = 0; n < norigins; n++)
			h_params[n] = startparams.centerparams;

		d_CTFFitCreateTarget2D(d_image, dimsimage, d_origins, h_params, norigins, fp, d_ps2dpolar, d_ps2dcoords, true, d_ps1dmin, d_ps1dmax);	// All averaged to one for background
		d_SubtractVector(d_ps1dmax, d_ps1dmin, d_ps1dmax, dimspolar.x);
		d_MaxOp(d_ps1dmax, 0.2f, d_ps1dmax, dimspolar.x);

		// Extract, average, convert to polar
		d_CTFPeriodogram(d_image, dimsimage, d_origins, norigins, fp.dimsperiodogram, d_ps2d);
		d_Cart2PolarFFT(d_ps2d, d_ps2dpolar, fp.dimsperiodogram, T_INTERP_CUBIC, fp.maskinnerradius, fp.maskouterradius, norigins);

		// Create polar background image
		tfloat* d_ps2dmin;
		cudaMalloc((void**)&d_ps2dmin, Elements2(dimspolar) * sizeof(tfloat));
		CudaMemcpyMulti(d_ps2dmin, d_ps1dmin, dimspolar.x, dimspolar.y, 1);
		//d_WriteMRC(d_ps2dmin, toInt3(dimspolar.x, dimspolar.y, 1), "d_ps2dmin.mrc");
		tfloat* d_ps2dmax;
		cudaMalloc((void**)&d_ps2dmax, Elements2(dimspolar) * sizeof(tfloat));
		CudaMemcpyMulti(d_ps2dmax, d_ps1dmax, dimspolar.x, dimspolar.y, 1);
		//d_WriteMRC(d_ps2dmax, toInt3(dimspolar.x, dimspolar.y, 1), "d_ps2dmax.mrc");

		// Subtract background and normalize
		d_SubtractVector(d_ps2dpolar, d_ps2dmin, d_ps2dpolar, Elements2(dimspolar), norigins);
		d_DivideSafeByVector(d_ps2dpolar, d_ps2dmax, d_ps2dpolar, Elements2(dimspolar), norigins);
		d_NormMonolithic(d_ps2dpolar, d_ps2dpolar, Elements2(dimspolar), T_NORM_MEAN01STD, norigins);

		// Create 1D targets and normalize
		d_CTFFitCreateTarget1D(d_ps2dpolar, d_ps2dcoords, dimspolar, h_params, norigins, fp, d_ps1d, d_ps1dcoords);
		d_ValueFill(d_ps2d, dimspolar.x, (tfloat)0);
		d_ValueFill(d_ps2d + originalfp.maskinnerradius, originalfp.maskouterradius - originalfp.maskinnerradius, (tfloat)1);
		CudaMemcpyMulti(d_ps2d, d_ps2d, dimspolar.x, norigins);
		d_NormMonolithic(d_ps1d, d_ps1d, dimspolar.x, d_ps2d, T_NORM_MEAN01STD, norigins);
		//d_WriteMRC(d_ps1d, toInt3(dimspolar.x, norigins, 1), "d_ps1d.mrc");

		cudaFree(d_ps2dmax);
		cudaFree(d_ps2dmin);
		cudaFree(d_ps2dcoords);
		cudaFree(d_ps2dpolar);
		cudaFree(d_ps2d);
		cudaFree(d_ps1dmax);
		cudaFree(d_ps1dmin);
		cudaFree(d_origins);

		free(h_params);
	}

	fp = originalfp;
	dimspolar.x = fp.maskouterradius - fp.maskinnerradius;

	// Store radius & angle for each 1D target point
	{
		float2* h_ps1dcoords = (float2*)malloc(dimspolar.x * sizeof(float2));
		float invhalfsize = 2.0f / (float)fp.dimsperiodogram.x;
		for (int r = 0; r < dimspolar.x; r++)
		{
			float rf = (float)(r + fp.maskinnerradius) * invhalfsize;
			h_ps1dcoords[r] = make_float2(rf, 0.0f);
		}
		cudaMemcpy(d_ps1dcoords, h_ps1dcoords, dimspolar.x * sizeof(float2), cudaMemcpyHostToDevice);
		free(h_ps1dcoords);
	}

	{
		for (int a = 0; a < v_angles.size(); a++)
		{
			CTFFitParams anglefp = fp;
			vector<pair<tfloat, CTFParams>> v_params;
			AddCTFParamsRange(v_params, anglefp);

			// Calculate defocus offsets across grid
			tfloat* d_griddefoci;
			{
				tfloat* h_griddefoci = (tfloat*)malloc(norigins * sizeof(tfloat));
				CTFTiltParams currenttilt(startparams.imageangle, startparams.stageangle, tfloat2(v_angles[a].x, v_angles[a].y), startparams.centerparams);
				currenttilt.GetZGrid2D(dimsimage, fp.dimsperiodogram, h_origins, norigins, h_griddefoci);
				d_griddefoci = (tfloat*)CudaMallocFromHostArray(h_griddefoci, norigins * sizeof(tfloat));
				free(h_griddefoci);
			}

			for (uint d = 0; d <= defocusrefinements; d++)
			{
				// Defocus search space
				tfloat* h_defocusoffsets = (tfloat*)malloc(v_params.size() * sizeof(tfloat));
				CTFParams* h_params = (CTFParams*)malloc(v_params.size() * sizeof(CTFParams));

				// Adjust copies to various defoci
				for (uint n = 0; n < v_params.size(); n++)
				{
					h_params[n] = startparams.centerparams;
					h_params[n].defocus += v_params[n].second.defocus;
					h_defocusoffsets[n] = v_params[n].second.defocus;
				}

				// Finally, accumulate the spectra based on the suggested defocus values
				tfloat* d_defocusoffsets = (tfloat*)CudaMallocFromHostArray(h_defocusoffsets, v_params.size() * sizeof(tfloat));
				tfloat* d_accumulated;
				cudaMalloc((void**)&d_accumulated, dimspolar.x * v_params.size() * sizeof(tfloat));
				d_AccumulateSpectra(d_ps1d, d_griddefoci, norigins, d_accumulated, startparams.centerparams.defocus, d_defocusoffsets, startparams.centerparams, fp, v_params.size());
				d_NormMonolithic(d_accumulated, d_accumulated, dimspolar.x, T_NORM_MEAN01STD, v_params.size());
				//CudaWriteToBinaryFile("d_accumulated.bin", d_accumulated, dimspolar.x * v_params.size() * sizeof(tfloat));
				cudaFree(d_defocusoffsets);
				free(h_defocusoffsets);

				// Simulate CTF
				tfloat* d_ctfsim;
				cudaMalloc((void**)&d_ctfsim, dimspolar.x * v_params.size() * sizeof(tfloat));
				d_CTFSimulate(h_params, d_ps1dcoords, d_ctfsim, dimspolar.x, true, v_params.size());
				d_NormMonolithic(d_ctfsim, d_ctfsim, dimspolar.x, T_NORM_MEAN01STD, v_params.size());
				//CudaWriteToBinaryFile("d_ctfsim.bin", d_ctfsim, dimspolar.x * v_params.size() * sizeof(tfloat));
				free(h_params);

				// Correlate
				d_MultiplyByVector(d_ctfsim, d_accumulated, d_ctfsim, dimspolar.x * v_params.size());
				d_SumMonolithic(d_ctfsim, d_accumulated, dimspolar.x, v_params.size());
				tfloat* h_scores = (tfloat*)MallocFromDeviceArray(d_accumulated, v_params.size() * sizeof(tfloat));
				for (uint n = 0; n < v_params.size(); n++)
					v_params[n].first = h_scores[n] / (tfloat)dimspolar.x;
				free(h_scores);
				cudaFree(d_ctfsim);
				cudaFree(d_accumulated);

				// Sort defoci by score in descending order
				sort(v_params.begin(), v_params.end(),
					[](const pair<tfloat, CTFFitParams> &a, const pair<tfloat, CTFFitParams> &b) -> bool
				{
					return a.first > b.first;
				});

				if (d < defocusrefinements)
				{
					vector<pair<tfloat, CTFParams>> v_newparams;
					anglefp.defocus.z /= 4.0;
					for (uint f = 0; f < 5; f++)
					{
						CTFFitParams localfp;
						localfp.defocus = tfloat3(v_params[f].second.defocus - anglefp.defocus.z * 3.0, 
												  v_params[f].second.defocus + anglefp.defocus.z * 3.0, 
												  anglefp.defocus.z);
						AddCTFParamsRange(v_newparams, localfp);
					}
					v_params = v_newparams;
				}
			}
			cudaFree(d_griddefoci);

			v_results.push_back(tfloat2(v_params[0].first, v_params[0].second.defocus));	
		}
	}
	
	cudaFree(d_ps1d);
	cudaFree(d_ps1dcoords);

	free(h_origins);
}