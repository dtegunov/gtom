#include "Prerequisites.cuh"
#include "CTF.cuh"
#include "FFT.cuh"
#include "Generics.cuh"
#include "Helper.cuh"
#include "ImageManipulation.cuh"
#include "Masking.cuh"
#include "Optimization.cuh"
#include "Transformation.cuh"

/////////////////////////////////////
//Fit specified parameters of a CTF//
/////////////////////////////////////

void d_CTFFitCreateTarget(tfloat* d_image, int2 dimsimage, tfloat* d_decay, int3* d_origins, int norigins, CTFFitParams p, tfloat* d_densetarget, float2* d_densecoords)
{
	tfloat* d_ps;
	cudaMalloc((void**)&d_ps, ElementsFFT2(p.dimsperiodogram) * sizeof(tfloat));
	d_Periodogram(d_image, dimsimage, d_origins, norigins, p.dimsperiodogram, d_ps);
	d_Log(d_ps, d_ps, ElementsFFT2(p.dimsperiodogram));

	int2 dimspolar = GetCart2PolarFFTSize(p.dimsperiodogram);
	tfloat* d_pspolar;
	cudaMalloc((void**)&d_pspolar, Elements2(dimspolar) * sizeof(tfloat));
	d_Cart2PolarFFT(d_ps, d_pspolar, p.dimsperiodogram, T_INTERP_CUBIC);

	int2 dimsps = toInt2(p.maskouterradius - p.maskinnerradius, dimspolar.y);
	for (int y = 0; y < dimsps.y; y++)
		cudaMemcpy(d_ps + dimsps.x * y, d_pspolar + dimspolar.x * y + p.maskinnerradius, dimsps.x * sizeof(tfloat), cudaMemcpyDeviceToDevice);
	cudaFree(d_pspolar);

	tfloat* d_background;
	if (d_decay == NULL)
	{
		cudaMalloc((void**)&d_background, Elements2(dimsps) * sizeof(tfloat));
		d_CTFDecay(d_ps, d_background, dimsps, 4, 16);
	}
	else
	{
		d_background = d_decay;
	}

	d_SubtractVector(d_ps, d_background, d_ps, Elements2(dimsps));

	if (d_decay == NULL)
		cudaFree(d_background);

	uint denselength = GetCart2PolarFFTNonredundantSize(p.dimsperiodogram, p.maskinnerradius, p.maskouterradius);
	float2* h_polar2dense = (float2*)malloc(denselength * sizeof(float2));

	for (int r = p.maskinnerradius, i = 0; r < p.maskouterradius; r++)
	{
		int steps = r * 2;
		float anglestep = (float)dimsps.y / (float)steps;
		for (int a = 0; a < steps; a++)
			h_polar2dense[i++] = make_float2((float)(r - p.maskinnerradius) + 0.5f, (float)a * anglestep + 0.5f);
	}
	float2* d_polar2dense = (float2*)CudaMallocFromHostArray(h_polar2dense, denselength * sizeof(float2));
	free(h_polar2dense);

	d_RemapInterpolated2D(d_ps, dimsps, d_densetarget, d_polar2dense, denselength, T_INTERP_CUBIC);
	d_Norm(d_densetarget, d_densetarget, denselength, (tfloat*)NULL, T_NORM_MEAN01STD, (tfloat)0);
	cudaFree(d_polar2dense);
	cudaFree(d_ps);

	float2* h_ctfpoints = (float2*)malloc(denselength * sizeof(float2));
	float invhalfsize = 2.0f / (float)p.dimsperiodogram.x;
	for (int r = p.maskinnerradius, i = 0; r < p.maskouterradius; r++)
	{
		float rf = (float)r;
		int steps = r * 2;
		float anglestep = PI / (float)steps;
		for (int a = 0; a < steps; a++)
		{
			float angle = (float)a * anglestep + PIHALF;
			float2 point = make_float2(cos(angle) * rf * invhalfsize, sin(angle) * rf * invhalfsize);
			h_ctfpoints[i++] = make_float2(sqrt(point.x * point.x + point.y * point.y), angle);
		}
	}
	cudaMemcpy(d_densecoords, h_ctfpoints, denselength * sizeof(float2), cudaMemcpyHostToDevice);
	free(h_ctfpoints);
}

void d_CTFFit(tfloat* d_dense, float2* d_densepoints, uint denselength, CTFFitParams p, int refinements, CTFParams &fit, tfloat &score, tfloat &mean, tfloat &stddev)
{
	tfloat* d_simulated;
	cudaMalloc((void**)&d_simulated, denselength * sizeof(tfloat));

	CTFParams bestfit;
	tfloat bestscore = 0.0f;
	vector<tfloat> scores;
	vector<pair<tfloat, CTFParams>> v_params;

	AddCTFParamsRange(v_params, p);

	for (int i = 0; i < refinements + 1; i++)
	{
		long memlimit = 128 * 1024 * 1024;
		int batchsize = min(32768, min(memlimit / (long)(denselength * sizeof(tfloat)), (int)v_params.size()));
		tfloat* d_batchsim;
		cudaMalloc((void**)&d_batchsim, denselength * batchsize * sizeof(tfloat));
		tfloat* d_batchscores;
		cudaMalloc((void**)&d_batchscores, v_params.size() * sizeof(tfloat));
		tfloat* h_batchscores = (tfloat*)malloc(v_params.size() * sizeof(tfloat));
		CTFParams* h_params = (CTFParams*)malloc(v_params.size() * sizeof(CTFParams));
		for (int i = 0; i < v_params.size(); i++)
			h_params[i] = v_params[i].second;

		for (int b = 0; b < v_params.size(); b += batchsize)
		{
			int curbatch = min((int)v_params.size() - b, batchsize);

			d_CTFSimulate(h_params + b, d_densepoints, d_batchsim, denselength, true, curbatch);
			d_NormMonolithic(d_batchsim, d_batchsim, denselength, (tfloat*)NULL, T_NORM_MEAN01STD, curbatch);
			d_MultiplyByVector(d_batchsim, d_dense, d_batchsim, denselength, curbatch);
			d_SumMonolithic(d_batchsim, d_batchscores + b, denselength, curbatch);
		}
		free(h_params);
		cudaMemcpy(h_batchscores, d_batchscores, v_params.size() * sizeof(tfloat), cudaMemcpyDeviceToHost);
		for (int i = 0; i < v_params.size(); i++)
		{
			h_batchscores[i] /= (tfloat)denselength;
			v_params[i].first = h_batchscores[i];
			scores.push_back(h_batchscores[i]);
		}
		free(h_batchscores);
		cudaFree(d_batchscores);
		cudaFree(d_batchsim);

		// Sort v_params by score in descending order
		sort(v_params.begin(), v_params.end(),
			[](const pair<tfloat, CTFFitParams> &a, const pair<tfloat, CTFFitParams> &b) -> bool
		{
			return a.first > b.first;
		});

		bestscore = v_params[0].first;
		bestfit = v_params[0].second;

		// Decrease search step size
		tfloat3* h_p = (tfloat3*)&p;
		for (int j = 0; j < 11; j++)
			if (h_p[j].x != h_p[j].y)
				h_p[j].z /= 4.0;

		vector<pair<tfloat, CTFParams>> v_paramsNew;
		for (int i = 0; i < min(10, (int)v_params.size()); i++)
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

	cudaFree(d_simulated);

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
	fit = bestfit;
	score = bestscore;
}

void d_CTFFit(tfloat* d_image, int2 dimsimage, int3* d_origins, int norigins, CTFFitParams p, int refinements, CTFParams &fit, tfloat &score, tfloat &mean, tfloat &stddev)
{
	uint denselength = GetCart2PolarFFTNonredundantSize(p.dimsperiodogram, p.maskinnerradius, p.maskouterradius);
	tfloat* d_ps;
	cudaMalloc((void**)&d_ps, denselength * sizeof(tfloat));
	float2* d_ctfpoints;
	cudaMalloc((void**)&d_ctfpoints, denselength * sizeof(float2));

	d_CTFFitCreateTarget(d_image, dimsimage, NULL, d_origins, norigins, p, d_ps, d_ctfpoints);

	d_CTFFit(d_ps, d_ctfpoints, denselength, p, refinements, fit, score, mean, stddev);
}

void AddCTFParamsRange(vector<pair<tfloat, CTFParams>> &v_params, CTFFitParams p)
{
	for (tfloat pixelsize = p.pixelsize.x; pixelsize <= p.pixelsize.y; pixelsize += p.pixelsize.z)
	{
		for (tfloat cs = p.Cs.x; cs <= p.Cs.y; cs += p.Cs.z)
		{
			for (tfloat cc = p.Cc.x; cc <= p.Cc.y; cc += p.Cc.z)
			{
				for (tfloat voltage = p.voltage.x; voltage <= p.voltage.y; voltage += p.voltage.z)
				{
					for (tfloat defocus = p.defocus.x; defocus <= p.defocus.y; defocus += p.defocus.z)
					{
						for (tfloat defocusdelta = p.defocusdelta.x; defocusdelta <= p.defocusdelta.y; defocusdelta += p.defocusdelta.z)
						{
							for (tfloat astigmatismangle = p.astigmatismangle.x; astigmatismangle <= p.astigmatismangle.y; astigmatismangle += p.astigmatismangle.z)
							{
								for (tfloat amplitude = p.amplitude.x; amplitude <= p.amplitude.y; amplitude += p.amplitude.z)
								{
									for (tfloat bfactor = p.Bfactor.x; bfactor <= p.Bfactor.y; bfactor += p.Bfactor.z)
									{
										for (tfloat decaycoh = p.decayCohIll.x; decaycoh <= p.decayCohIll.y; decaycoh += p.decayCohIll.z)
										{
											for (tfloat decayspread = p.decayspread.x; decayspread <= p.decayspread.y; decayspread += p.decayspread.z)
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

												v_params.push_back(pair<tfloat, CTFParams>((tfloat)0, testparams));

												if (p.decayspread.x == p.decayspread.y)
													break;
											}
											if (p.decayCohIll.x == p.decayCohIll.y)
												break;
										}
										if (p.Bfactor.x == p.Bfactor.y)
											break;
									}
									if (p.amplitude.x == p.amplitude.y)
										break;
								}
								if (p.astigmatismangle.x == p.astigmatismangle.y || defocusdelta == 0.0f)
									break;
							}
							if (p.defocusdelta.x == p.defocusdelta.y)
								break;
						}
						if (p.defocus.x == p.defocus.y)
							break;
					}
					if (p.voltage.x == p.voltage.y)
						break;
				}
				if (p.Cc.x == p.Cc.y)
					break;
			}
			if (p.Cs.x == p.Cs.y)
				break;
		}
		if (p.pixelsize.x == p.pixelsize.y)
			break;
	}
}