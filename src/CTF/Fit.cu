#include "Prerequisites.cuh"
#include "Correlation.cuh"
#include "CTF.cuh"
#include "FFT.cuh"
#include "Generics.cuh"
#include "Helper.cuh"
#include "ImageManipulation.cuh"
#include "Masking.cuh"
#include "Optimization.cuh"
#include "PCA.cuh"
#include "Transformation.cuh"


namespace gtom
{
	void h_CTFFitEnvelope(tfloat* h_input, uint diminput, tfloat* h_envelopemin, tfloat* h_envelopemax, char peakextent, uint outputstart, uint outputend, uint batch)
	{
		// This function does not consider the zero frequency, i. e. starts counting at 1
		outputend = min(outputend, diminput - 1);
		uint outputlength = outputend - outputstart + 1;
		for (uint b = 0; b < batch; b++)
		{
			// Locate maxima and minima
			tfloat* h_temp = h_input + b * diminput;
			std::vector<tfloat2> minima;
			std::vector<tfloat2> maxima;
			for (int i = 1; i < diminput; i++)
			{
				tfloat refval = h_temp[i];
				bool ismin = true, ismax = true;
				for (int j = max(1, i - peakextent); j <= min(diminput - 1, i + peakextent); j++)
				{
					if (h_temp[j] > refval)
						ismax = false;
					else if (h_temp[j] < refval)
						ismin = false;
				}
				if (ismin == ismax)
					continue;
				else if (ismin)
					minima.push_back(tfloat2(i, refval));
				else
					maxima.push_back(tfloat2(i, refval));
			}

			tfloat* h_tempmin = h_envelopemin + b * outputlength;
			tfloat* h_tempmax = h_envelopemax + b * outputlength;
			// When no peaks found, make absolut max/min values the envelope (i. e. 2 horizontal lines)
			if (minima.size() == 0 || maxima.size() == 0)
			{
				tfloat minval = 1e30, maxval = -1e30;
				for (uint i = 1; i < diminput; i++)
				{
					minval = min(minval, h_temp[i]);
					maxval = max(maxval, h_temp[i]);
				}
				for (uint i = 0; i < outputlength; i++)
				{
					h_tempmin[i] = minval;
					h_tempmax[i] = maxval;
				}
				continue;
			}

			// Good to interpolate
			Interpolate1DOntoGrid(minima, h_tempmin, outputstart, outputend);
			Interpolate1DOntoGrid(maxima, h_tempmax, outputstart, outputend);
		}
	}

	void d_CTFFitCreateTarget2D(tfloat* d_image, int2 dimsimage, CTFParams params, CTFFitParams fp, float overlapfraction, tfloat* d_ps2dpolar, float2* d_ps2dcoords)
	{
		int2 dimsregion = fp.dimsperiodogram;

		// Create uniform grid over the image
		int2 regions;
		int3* h_origins = GetEqualGridSpacing(dimsimage, dimsregion, overlapfraction, regions);
		int3* d_origins = (int3*)CudaMallocFromHostArray(h_origins, Elements2(regions) * sizeof(int3));
		free(h_origins);

		int norigins = Elements2(regions);

		CTFParams* h_params = (CTFParams*)malloc(norigins * sizeof(CTFParams));
		for (uint n = 0; n < norigins; n++)
			h_params[n] = params;

		// Call the custom-grid version to extract 2D spectra
		d_CTFFitCreateTarget2D(d_image, dimsimage, d_origins, h_params, Elements2(regions), fp, d_ps2dpolar, d_ps2dcoords, true);

		cudaFree(d_origins);
		free(h_params);
	}

	void d_CTFFitCreateTarget2D(tfloat* d_image, int2 dimsimage, int3* d_origins, CTFParams* h_params, int norigins, CTFFitParams fp, tfloat* d_ps2dpolar, float2* d_ps2dcoords, bool sumtoone, tfloat* d_outps1dmin, tfloat* d_outps1dmax)
	{
		// Create raw power spectra in cartesian coords
		tfloat* d_ps2d;
		cudaMalloc((void**)&d_ps2d, ElementsFFT2(fp.dimsperiodogram) * norigins * sizeof(tfloat));
		d_CTFPeriodogram(d_image, dimsimage, d_origins, norigins, fp.dimsperiodogram, d_ps2d);
		if (sumtoone)
		{
			d_ReduceMean(d_ps2d, d_ps2d, ElementsFFT2(fp.dimsperiodogram), norigins);
			norigins = 1;
		}
		//d_WriteMRC(d_ps2d, toInt3(ElementsFFT1(fp.dimsperiodogram.x), fp.dimsperiodogram.y, norigins), "d_ps2d.mrc");

		// Create 1D radial averages
		uint length1d = fp.dimsperiodogram.x / 2;
		tfloat* d_ps1d;
		cudaMalloc((void**)&d_ps1d, length1d * norigins * sizeof(tfloat));
		d_CTFRotationalAverage(d_ps2d, fp.dimsperiodogram, h_params, d_ps1d, 0, length1d, norigins);
		//CudaWriteToBinaryFile("d_ps1d.bin", d_ps1d, length1d * norigins * sizeof(tfloat));

		if (norigins > 1)
			d_PCAFilter(d_ps1d, length1d, norigins, 5, d_ps1d);
		//CudaWriteToBinaryFile("d_ps1d.bin", d_ps1d, length1d * norigins * sizeof(tfloat));

		// Fit envelopes to 1D radial averages
		uint relevantlength1d = fp.maskouterradius - fp.maskinnerradius;
		tfloat* h_ps1d = (tfloat*)MallocFromDeviceArray(d_ps1d, length1d * norigins * sizeof(tfloat));
		tfloat* h_ps1dmin = (tfloat*)malloc(relevantlength1d * norigins * sizeof(tfloat));
		tfloat* h_ps1dmax = (tfloat*)malloc(relevantlength1d * norigins * sizeof(tfloat));
		h_CTFFitEnvelope(h_ps1d, length1d, h_ps1dmin, h_ps1dmax, max(2, length1d / 32), fp.maskinnerradius, fp.maskouterradius - 1, norigins);
		tfloat* d_ps1dmin = (tfloat*)CudaMallocFromHostArray(h_ps1dmin, relevantlength1d * norigins * sizeof(tfloat));
		tfloat* d_ps1dmax = (tfloat*)CudaMallocFromHostArray(h_ps1dmax, relevantlength1d * norigins * sizeof(tfloat));
		//CudaWriteToBinaryFile("d_ps1dmin.bin", d_ps1dmin, relevantlength1d * norigins * sizeof(tfloat));
		//CudaWriteToBinaryFile("d_ps1dmax.bin", d_ps1dmax, relevantlength1d * norigins * sizeof(tfloat));
		free(h_ps1dmax);
		free(h_ps1dmin);
		free(h_ps1d);
		cudaFree(d_ps1d);

		// PCA-filter the envelopes to first 5 PCs if norigins > 1
		if (norigins > 1)
			d_PCAFilter(d_ps1dmin, relevantlength1d, norigins, min(norigins, 2), d_ps1dmin);
		if (d_outps1dmin != NULL)
			cudaMemcpy(d_outps1dmin, d_ps1dmin, relevantlength1d * norigins * sizeof(tfloat), cudaMemcpyDeviceToDevice);
		if (d_outps1dmax != NULL)
			cudaMemcpy(d_outps1dmax, d_ps1dmax, relevantlength1d * norigins * sizeof(tfloat), cudaMemcpyDeviceToDevice);
		//CudaWriteToBinaryFile("d_ps1dmin.bin", d_ps1dmin, relevantlength1d * norigins * sizeof(tfloat));

		// Convert 2D power spectra to polar coords
		int2 dimspolar = GetCart2PolarFFTSize(fp.dimsperiodogram);
		dimspolar.x = min(fp.dimsperiodogram.x / 2, fp.maskouterradius - fp.maskinnerradius);
		d_Cart2PolarFFT(d_ps2d, d_ps2dpolar, fp.dimsperiodogram, T_INTERP_CUBIC, fp.maskinnerradius, fp.maskouterradius, norigins);
		cudaFree(d_ps2d);
		//d_WriteMRC(d_ps2dpolar, toInt3(dimspolar.x, dimspolar.y, norigins), "d_ps2dpolar.mrc");

		// Create polar background image
		tfloat* d_ps2dmin;
		cudaMalloc((void**)&d_ps2dmin, Elements2(dimspolar) * norigins * sizeof(tfloat));
		CudaMemcpyMulti(d_ps2dmin, d_ps1dmin, relevantlength1d, dimspolar.y, norigins);
		cudaFree(d_ps1dmin);
		cudaFree(d_ps1dmax);
		//d_WriteMRC(d_ps2dmin, toInt3(dimspolar.x, dimspolar.y, norigins), "d_ps2dmin.mrc");

		// Subtract background, PCA-filter, and normalize everyone
		d_SubtractVector(d_ps2dpolar, d_ps2dmin, d_ps2dpolar, Elements2(dimspolar) * norigins);
		/*if (norigins > 1)
			d_PCAFilter(d_ps2dpolar, Elements2(dimspolar), norigins, norigins / 10, d_ps2dpolar);*/
		d_NormMonolithic(d_ps2dpolar, d_ps2dpolar, Elements2(dimspolar), T_NORM_MEAN01STD, norigins);
		cudaFree(d_ps2dmin);
		//d_WriteMRC(d_ps2dpolar, toInt3(dimspolar.x, dimspolar.y, norigins), "d_ps2dpolar.mrc");

		// Store radius & angle for each target point
		float2* h_ps2dcoords = (float2*)malloc(Elements2(dimspolar) * sizeof(float2));
		float invhalfsize = 2.0f / (float)fp.dimsperiodogram.x;
		float anglestep = PI / (float)(dimspolar.y - 1);
		for (int a = 0; a < dimspolar.y; a++)
		{
			float angle = (float)a * anglestep + PIHALF;
			for (int r = 0; r < dimspolar.x; r++)
				h_ps2dcoords[a * dimspolar.x + r] = make_float2((r + fp.maskinnerradius) * invhalfsize, angle);
		}
		cudaMemcpy(d_ps2dcoords, h_ps2dcoords, Elements2(dimspolar) * sizeof(float2), cudaMemcpyHostToDevice);
		free(h_ps2dcoords);
	}

	void d_CTFFitCreateTarget1D(tfloat* d_ps2dpolar, float2* d_ps2dcoords, int2 dimspolar, CTFParams* h_params, int norigins, CTFFitParams fp, tfloat* d_ps1d, float2* d_ps1dcoords)
	{
		// Create radial averages based on 2D targets in polar coords
		d_CTFRotationalAverage(d_ps2dpolar, d_ps2dcoords, Elements2(dimspolar), fp.dimsperiodogram.x, h_params, d_ps1d, fp.maskinnerradius, fp.maskouterradius, norigins);
		/*if (norigins > 1)
			d_PCAFilter(d_ps1d, dimspolar.x, norigins, min(norigins / 10, dimspolar.x / 5), d_ps1d);*/
		d_NormMonolithic(d_ps1d, d_ps1d, fp.maskouterradius - fp.maskinnerradius, T_NORM_MEAN01STD, norigins);
		//CudaWriteToBinaryFile("d_ps1d.bin", d_ps1d, dimspolar.x * norigins * sizeof(tfloat));

		// Store radius & angle for each 1D target point
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

	void d_CTFFit(tfloat* d_target, float2* d_targetcoords, int2 dimstarget, CTFParams* h_startparams, uint ntargets, CTFFitParams p, int refinements, std::vector<std::pair<tfloat, CTFParams> > &fits, tfloat &score, tfloat &mean, tfloat &stddev)
	{
		uint targetlength = Elements2(dimstarget);

		tcomplex* d_targetft;
		if (dimstarget.y > 1)
		{
			cudaMalloc((void**)&d_targetft, ElementsFFT2(dimstarget) * ntargets * sizeof(tcomplex));
			d_FFTR2C(d_target, d_targetft, 2, toInt3(dimstarget), ntargets);
		}

		CTFParams bestfit;
		tfloat bestscore = 0.0f;
		std::vector<tfloat> scores;
		std::vector<std::pair<tfloat, CTFParams> > v_params;

		AddCTFParamsRange(v_params, p);

		for (int i = 0; i < refinements + 1; i++)
		{
			int memlimit = 512 << 20;
			int batchsize = min(32768, min((int)(memlimit / (targetlength * ntargets * sizeof(tfloat))), (int)v_params.size()));

			tfloat* d_batchsim;
			cudaMalloc((void**)&d_batchsim, targetlength * ntargets * batchsize * sizeof(tfloat));
			tcomplex* d_batchsimft;
			tfloat3* d_peakpos;
			if (dimstarget.y > 1)
			{
				cudaMalloc((void**)&d_batchsimft, ElementsFFT2(dimstarget) * batchsize * ntargets * sizeof(tcomplex));
				cudaMalloc((void**)&d_peakpos, batchsize * sizeof(tfloat3));
			}
			tfloat* d_targetscores = (tfloat*)CudaMallocValueFilled(batchsize * ntargets, (tfloat)0);
			tfloat* d_batchscores = (tfloat*)CudaMallocValueFilled(v_params.size(), (tfloat)0);
			tfloat* h_batchscores = (tfloat*)malloc(v_params.size() * sizeof(tfloat));
			CTFParams* h_params = (CTFParams*)malloc(batchsize * ntargets * sizeof(CTFParams));

			for (int b = 0; b < v_params.size(); b += batchsize)
			{
				int curbatch = min((int)v_params.size() - b, batchsize);
				// For every group of targets in current batch...
				for (uint i = 0; i < curbatch; i++)
				{
					// ... take the group's adjustment calculated based on CTFFitParams...
					CTFParams adjustment = v_params[b + i].second;
					for (uint n = 0; n < ntargets; n++)
					{
						// ... and add it to the original CTFParams of each target.
						CTFParams* original = h_startparams + n;
						CTFParams* adjusted = h_params + i * ntargets + n;
						for (uint p = 0; p < 11; p++)
							((tfloat*)adjusted)[p] = ((tfloat*)original)[p] + ((tfloat*)&adjustment)[p];

						// Angle has already been taken into account when creating rotational average, thus set to 0
						if (dimstarget.y == 1)
							h_params[i * ntargets + n].astigmatismangle = 0;
					}
				}

				// Simulate the batch's CTF based on adjusted params, and normalize for CC
				d_CTFSimulate(h_params, d_targetcoords, d_batchsim, targetlength, true, curbatch * ntargets);
				d_NormMonolithic(d_batchsim, d_batchsim, targetlength, T_NORM_MEAN01STD, curbatch * ntargets);
				//d_WriteMRC(d_batchsim, toInt3(dimstarget.x, dimstarget.y, curbatch * ntargets), "d_batchsim.mrc");

				// Working with 1D averages only, no need to determine astigmatism angle
				if (dimstarget.y == 1)
				{
					// CC every group of simulated spectra with the original n targets
					d_MultiplyByVector(d_batchsim, d_target, d_batchsim, targetlength * ntargets, curbatch);
					d_SumMonolithic(d_batchsim, d_targetscores, targetlength, curbatch * ntargets);

					// Sum up CC for each group
					d_ReduceMean(d_targetscores, d_batchscores + b, 1, ntargets, curbatch);
				}
				else
				{
					// Perform CC through FFT
					d_FFTR2C(d_batchsim, d_batchsimft, 2, toInt3(dimstarget), curbatch * ntargets);
					d_ComplexMultiplyByConjVector(d_batchsimft, d_targetft, d_batchsimft, ElementsFFT2(dimstarget) * ntargets, curbatch);
					d_IFFTC2R(d_batchsimft, d_batchsim, 2, toInt3(dimstarget), ntargets * curbatch, false);
					d_RemapFullFFT2Full(d_batchsim, (tfloat*)d_batchsimft, toInt3(dimstarget), ntargets * curbatch);
					//d_WriteMRC((tfloat*)d_batchsimft, toInt3(dimstarget.x, dimstarget.y, ntargets * curbatch), "d_batchsimcorr.mrc");

					// Extract lines along x = 0 and find the peaks
					d_Extract((tfloat*)d_batchsimft, d_batchsim, toInt3(dimstarget), toInt3(1, dimstarget.y, 1), toInt3(dimstarget.x / 2, dimstarget.y / 2, 0), ntargets * curbatch);
					d_ReduceMean(d_batchsim, (tfloat*)d_batchsimft, dimstarget.y, ntargets, curbatch);
					//d_WriteMRC((tfloat*)d_batchsimft, toInt3(dimstarget.y, ntargets * curbatch, 1), "d_batchsimcorrextract.mrc");
					d_Peak((tfloat*)d_batchsimft, d_peakpos, d_batchscores + b, toInt3(dimstarget.y, 1, 1), T_PEAK_SUBCOARSE, NULL, NULL, curbatch);

					// Update astigmatism angles in fits
					tfloat3* h_peakpos = (tfloat3*)MallocFromDeviceArray(d_peakpos, curbatch * sizeof(tfloat3));
					for (uint i = 0; i < curbatch; i++)
						v_params[b + i].second.astigmatismangle = -(h_peakpos[i].x - dimstarget.y / 2) / (tfloat)dimstarget.y * PI;
					free(h_peakpos);
				}
			}
			free(h_params);
			cudaMemcpy(h_batchscores, d_batchscores, v_params.size() * sizeof(tfloat), cudaMemcpyDeviceToHost);
			// Normalize and assign CC values to their respective CTFParams
			for (int j = 0; j < v_params.size(); j++)
			{
				h_batchscores[j] /= dimstarget.y > 1 ? (tfloat)targetlength * (tfloat)targetlength : (tfloat)targetlength;
				v_params[j].first = h_batchscores[j];
				if (i == 0)
					scores.push_back(h_batchscores[j]);
			}
			free(h_batchscores);
			cudaFree(d_batchscores);
			cudaFree(d_targetscores);
			if (dimstarget.y > 1)
			{
				cudaFree(d_peakpos);
				cudaFree(d_batchsimft);
			}
			cudaFree(d_batchsim);

			// Sort v_params by score in descending order
			std::sort(v_params.begin(), v_params.end(),
				[](const std::pair<tfloat, CTFFitParams> &a, const std::pair<tfloat, CTFFitParams> &b) -> bool
			{
				return a.first > b.first;
			});

			bestscore = v_params[0].first;
			bestfit = v_params[0].second;

			if (i == refinements)
				break;

			// Decrease search step size
			tfloat3* h_p = (tfloat3*)&p;
			for (int j = 0; j < 11; j++)
				if (h_p[j].x != h_p[j].y)
					h_p[j].z /= 4.0;

			// Create CTFParams around the 5 best matches of this iteration, to be explored in the next one
			std::vector<std::pair<tfloat, CTFParams> > v_paramsNew;
			for (int i = 0; i < min(5, (int)v_params.size()); i++)
			{
				CTFParams fit = v_params[i].second;
				CTFFitParams pNew = p;
				tfloat3* h_p = (tfloat3*)&pNew;
				tfloat* h_f = (tfloat*)&fit;
				for (int j = 0; j < 11; j++)
					if (h_p[j].x != h_p[j].y)
						h_p[j] = tfloat3(h_f[j] - h_p[j].z * 3.0, h_f[j] + h_p[j].z * 3.0, h_p[j].z);
				AddCTFParamsRange(v_paramsNew, pNew);
			}

			v_params = v_paramsNew;
		}

		if (dimstarget.y > 1)
			cudaFree(d_targetft);

		if (scores.size() > 1)
		{
			mean = 0;
			for (int i = 0; i < scores.size(); i++)
				mean += scores[i];
			mean /= (tfloat)scores.size();
			stddev = 0;
			for (int i = 0; i < scores.size(); i++)
				stddev += pow(scores[i] - mean, 2.0);
			stddev = sqrt(stddev / (tfloat)scores.size());
		}
		for (int i = 0; i < v_params.size(); i++)
			fits.push_back(v_params[i]);
		score = bestscore;
	}

	void d_CTFFit(tfloat* d_image, int2 dimsimage, float overlapfraction, CTFParams startparams, CTFFitParams fp, int refinements, CTFParams &fit, tfloat &score, tfloat &mean, tfloat &stddev)
	{
		int2 polardims = GetCart2PolarFFTSize(fp.dimsperiodogram);
		polardims.x = fp.maskouterradius - fp.maskinnerradius;

		tfloat* d_ps2dpolar;
		cudaMalloc((void**)&d_ps2dpolar, Elements2(polardims) * sizeof(tfloat));
		float2* d_ps2dcoords;
		cudaMalloc((void**)&d_ps2dcoords, Elements2(polardims) * sizeof(float2));

		d_CTFFitCreateTarget2D(d_image, dimsimage, startparams, fp, overlapfraction, d_ps2dpolar, d_ps2dcoords);

		std::vector<std::pair<tfloat, CTFParams> > fits;
		d_CTFFit(d_ps2dpolar, d_ps2dcoords, polardims, &startparams, 1, fp, refinements, fits, score, mean, stddev);

		fit = fits[0].second;
	}

	void AddCTFParamsRange(std::vector<std::pair<tfloat, CTFParams> > &v_params, CTFFitParams p)
	{
		for (tfloat pixelsize = p.pixelsize.x; pixelsize <= p.pixelsize.y; pixelsize += max(1e-30, p.pixelsize.z))
			for (tfloat cs = p.Cs.x; cs <= p.Cs.y; cs += max(1e-30, p.Cs.z))
				for (tfloat cc = p.Cc.x; cc <= p.Cc.y; cc += max(1e-30, p.Cc.z))
					for (tfloat voltage = p.voltage.x; voltage <= p.voltage.y; voltage += max(1e-30, p.voltage.z))
						for (tfloat defocus = p.defocus.x; defocus <= p.defocus.y; defocus += max(1e-30, p.defocus.z))
							for (tfloat defocusdelta = p.defocusdelta.x; defocusdelta <= p.defocusdelta.y; defocusdelta += max(1e-30, p.defocusdelta.z))
								for (tfloat astigmatismangle = p.astigmatismangle.x; astigmatismangle <= p.astigmatismangle.y; astigmatismangle += max(1e-30, p.astigmatismangle.z))
									for (tfloat amplitude = p.amplitude.x; amplitude <= p.amplitude.y; amplitude += max(1e-30, p.amplitude.z))
										for (tfloat bfactor = p.Bfactor.x; bfactor <= p.Bfactor.y; bfactor += max(1e-30, p.Bfactor.z))
											for (tfloat decaycoh = p.decayCohIll.x; decaycoh <= p.decayCohIll.y; decaycoh += max(1e-30, p.decayCohIll.z))
												for (tfloat decayspread = p.decayspread.x; decayspread <= p.decayspread.y; decayspread += max(1e-30, p.decayspread.z))
												{
													CTFParams testparams;
													testparams.pixelsize = pixelsize;
													testparams.Cs = cs;
													testparams.Cc = cc;
													testparams.voltage = voltage;
													testparams.defocus = defocus;
													testparams.defocusdelta = defocusdelta;
													testparams.astigmatismangle = astigmatismangle;
													testparams.amplitude = amplitude;
													testparams.Bfactor = bfactor;
													testparams.decayCohIll = decaycoh;
													testparams.decayspread = decayspread;

													v_params.push_back(std::pair<tfloat, CTFParams>((tfloat)0, testparams));
												}
	}
}