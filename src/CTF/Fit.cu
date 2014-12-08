#include "Prerequisites.cuh"
#include "CTF.cuh"
#include "FFT.cuh"
#include "Generics.cuh"
#include "Helper.cuh"
#include "ImageManipulation.cuh"
#include "Masking.cuh"
#include "Optimization.cuh"
#include "Transformation.cuh"

//////////////////////////////////////////////
//Fit specified parameters of a CTF function//
//////////////////////////////////////////////

class CTFOptimizer : public Optimizer
{
public:
	typedef ::column_vector column_vector;
	typedef dlib::matrix<double> general_matrix;

	float2* d_x;
	tfloat* d_y;
	tfloat* d_sim;
	tfloat* d_r;
	tfloat* h_r;
	mutable column_vector r;
	tfloat* d_jac;
	tfloat* h_jac;
	uint denselength;
	CTFFitParams params;
	CTFParams fit;

	CTFOptimizer(tfloat* _d_ps,
				 float2* _d_ctfpoints,
				 uint _denselength,
				 CTFFitParams _params,
				 CTFParams _fit)
	{
		d_y = _d_ps;
		d_x = _d_ctfpoints;
		denselength = _denselength;
		params = _params;
		fit = _fit;
		cudaMalloc((void**)&d_sim, denselength * sizeof(tfloat));
		cudaMalloc((void**)&d_r, denselength * sizeof(tfloat));
		h_r = (tfloat*)malloc(denselength * sizeof(tfloat));
		r = column_vector(denselength);
		cudaMalloc((void**)&d_jac, denselength * 3 * sizeof(tfloat));
		h_jac = (tfloat*)malloc(denselength * 3 * sizeof(tfloat));
	}

	~CTFOptimizer()
	{
		free(h_jac);
		cudaFree(d_jac);
		free(h_r);
		cudaFree(d_r);
		cudaFree(d_sim);
	}

	CTFParams Vec2Params(const column_vector vec) const
	{
		CTFParams p = fit;
		/*p.pixelsize = vec(0);
		p.Cs = vec(1);
		p.Cc = vec(2);
		p.voltage = vec(3);
		p.defocus = vec(4);
		p.defocusdelta = vec(5);
		p.astigmatismangle = vec(6);
		p.amplitude = vec(7);
		p.Bfactor = vec(8);
		p.decayCohIll = vec(9);
		p.decayspread = vec(10);*/
		p.defocus = vec(0);
		p.astigmatismangle = vec(1);
		p.defocusdelta = vec(2);

		return p;
	}

	column_vector Params2Vec(const CTFParams p) const
	{
		column_vector vec = column_vector(3);
		/*vec(0) = p.pixelsize;
		vec(1) = p.Cs;
		vec(2) = p.Cc;
		vec(3) = p.voltage;
		vec(4) = p.defocus;
		vec(5) = p.defocusdelta;
		vec(6) = p.astigmatismangle;
		vec(7) = p.amplitude;
		vec(8) = p.decayKsquared;
		vec(9) = p.decayCohIll;
		vec(10) = p.decayspread;*/
		vec(0) = p.defocus;
		vec(1) = p.astigmatismangle;
		vec(2) = p.defocusdelta;

		return vec;
	}

	void simulate(const column_vector& params, tfloat* d_output) const
	{
		CTFParams testparams = Vec2Params(params);

		d_CTFSimulate(testparams, d_x, d_output, denselength, true);
		d_Norm(d_output, d_output, denselength, (tfloat*)NULL, T_NORM_MEAN01STD, 0);
	}

	//double operator() (const column_vector& params) const
	double Evaluate(column_vector &params)
	{
		simulate(params, d_sim);
		d_SubtractVector(d_y, d_sim, d_r, denselength);
		cudaMemcpy(h_r, d_r, denselength * sizeof(tfloat), cudaMemcpyDeviceToHost);
		for (int i = 0; i < denselength; i++)
			r(i) = h_r[i];

		d_MultiplyByVector(d_r, d_r, d_r, denselength);
		d_Sum(d_r, d_sim, denselength);
		tfloat score;
		cudaMemcpy(&score, d_sim, sizeof(tfloat), cudaMemcpyDeviceToHost);

		return sqrt(score / (double)denselength) * 100.0;
	}

	//void get_derivative_and_hessian(const column_vector& x, column_vector& der, general_matrix& hess) const
	column_vector Derivative(column_vector x)
	{
		general_matrix jac = general_matrix(denselength, 3);
		column_vector xcopy = column_vector(x.size());
		for (int i = 0; i < 3; i++)
			xcopy(i) = x(i);

		tfloat3* h_params = (tfloat3*)&params;
		h_params += 4;

		for (int i = 0; i < 3; i++)
			if (h_params[i].x != h_params[i].y)
			{
				double psi = h_params[i].z / 2.0;
				double old = xcopy(i);
				xcopy(i) = old + psi;
				this->simulate(xcopy, d_sim);
				xcopy(i) = old - psi;
				this->simulate(xcopy, d_r);
				xcopy(i) = old;

				d_SubtractVector(d_sim, d_r, d_r, denselength);
				d_DivideByScalar(d_r, d_r, denselength, 2.0f * (tfloat)psi);
				cudaMemcpy(h_r, d_r, denselength * sizeof(tfloat), cudaMemcpyDeviceToHost);
				for (int j = 0; j < denselength; j++)
					jac(j, i) = h_r[j] / (tfloat)denselength;
			}
			else
				for (int j = 0; j < denselength; j++)
					jac(j, i) = 0.0;

		/*this->simulate(x, d_sim);
		imgstats5* d_stats;
		cudaMalloc((void**)&d_stats, sizeof(imgstats5));
		d_Dev(d_sim, d_stats, denselength, (tfloat*)NULL);
		imgstats5 stats;
		cudaMemcpy(&stats, d_stats, sizeof(imgstats5), cudaMemcpyDeviceToHost);
		cudaFree(d_stats);

		CTFParams testparams = Vec2Params(x);
		d_CTFSimulateDerivative(testparams, d_x, d_jac, denselength);
		d_DivideByScalar(d_jac, d_jac, denselength * 3, stats.stddev * (tfloat)denselength);
		cudaMemcpy(h_jac, d_jac, denselength * 3 * sizeof(tfloat), cudaMemcpyDeviceToHost);
		for (int i = 0; i < denselength; i++)
			for (int j = 0; j < 3; j++)
				jac(i, j) = h_jac[j * denselength + i];*/

		return dlib::trans(jac) * r;
		//hess = dlib::trans(jac) * jac;
	}

	CTFParams Optimize(CTFParams start)
	{
		column_vector vecstart = Params2Vec(start);

		/*ofstream logfile;
		logfile.open(((string)("log.txt")).c_str(), ios::out);

		for (float angle = params.astigmatismangle.x; angle <= params.astigmatismangle.y; angle += params.astigmatismangle.z / 10.0f)
		for (float defocus = params.defocus.x; defocus <= params.defocus.y; defocus += params.defocus.z / 10.0f)
		{
			CTFParams p = fit;
			p.defocus = defocus;
			p.astigmatismangle = angle;
			logfile << this->Evaluate(Params2Vec(p)) << "\n";
			logfile.flush();
		}
		logfile.close();*/

		/*dlib::find_min_trust_region(dlib::objective_delta_stop_strategy(1e-4),
									*this,
									vecstart, 1000.0);*/
		dlib::find_min(dlib::bfgs_search_strategy(),
			dlib::objective_delta_stop_strategy(1e-4),
			EvalWrapper(this),
			DerivativeWrapper(this),
			vecstart,
			-1);

		return Vec2Params(vecstart);
	}
};

void d_CTFFit(tfloat* d_image, int2 dimsimage, int3* d_origins, int norigins, CTFFitParams p, int refinements, CTFParams &fit, tfloat &score, tfloat &mean, tfloat &stddev)
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
	cudaMalloc((void**)&d_background, Elements2(dimsps) * sizeof(tfloat));
	d_CTFDecay(d_ps, d_background, dimsps, 4, 8);
	d_SubtractVector(d_ps, d_background, d_ps, Elements2(dimsps));

	uint denselength = GetCart2PolarFFTNonredundantSize(p.dimsperiodogram, p.maskinnerradius, p.maskouterradius);
	float2* h_polar2dense = (float2*)malloc(denselength * sizeof(float2));
	float2* h_polar2densetemp = h_polar2dense;
	for (int r = p.maskinnerradius; r < p.maskouterradius; r++)
	{
		int steps = r * 2;
		float anglestep = (float)dimsps.y / (float)steps;
		for (int a = 0; a < steps; a++)
			*h_polar2densetemp++ = make_float2((float)(r - p.maskinnerradius) + 0.5f, (float)a * anglestep + 0.5f);
	}
	denselength = h_polar2densetemp - h_polar2dense;
	float2* d_polar2dense = (float2*)CudaMallocFromHostArray(h_polar2dense, denselength * sizeof(float2));
	free(h_polar2dense);

	d_RemapInterpolated2D(d_ps, dimsps, d_ps, d_polar2dense, denselength, T_INTERP_CUBIC);
	d_Norm(d_ps, d_ps, denselength, (tfloat*)NULL, T_NORM_MEAN01STD, (tfloat)0);
	cudaFree(d_polar2dense);

	float2* h_ctfpoints = (float2*)malloc(denselength * sizeof(float2));
	float2* h_ctfpointstemp = h_ctfpoints;
	float invhalfsize = 2.0f / (float)p.dimsperiodogram.x;
	for (int r = p.maskinnerradius; r < p.maskouterradius; r++)
	{
		float rf = (float)r;
		int steps = r * 2;
		float anglestep = PI / (float)steps;
		for (int a = 0; a < steps; a++)
		{
			float angle = (float)a * anglestep + PIHALF;
			float2 point = make_float2(cos(angle) * rf * invhalfsize, sin(angle) * rf * invhalfsize);
			*h_ctfpointstemp++ = make_float2(sqrt(point.x * point.x + point.y * point.y), angle);
		}
	}
	float2* d_ctfpoints = (float2*)CudaMallocFromHostArray(h_ctfpoints, denselength * sizeof(float2));
	free(h_ctfpoints);

	tfloat* d_simulated;
	cudaMalloc((void**)&d_simulated, denselength * sizeof(tfloat));

	CTFParams bestfit;
	tfloat bestscore = 0.0f;
	vector<tfloat> scores;

	for (int i = 0; i < refinements + 1; i++)
	{
		vector<CTFParams> v_params;

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

													v_params.push_back(testparams);

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

		int batchsize = min(192, (int)v_params.size());
		tfloat* d_batchsim;
		cudaMalloc((void**)&d_batchsim, denselength * batchsize * sizeof(tfloat));
		tfloat* d_batchscores;
		cudaMalloc((void**)&d_batchscores, v_params.size() * sizeof(tfloat));
		tfloat* h_batchscores = (tfloat*)malloc(v_params.size() * sizeof(tfloat));

		for (int b = 0; b < v_params.size(); b += batchsize)
		{
			int curbatch = min((int)v_params.size() - b, batchsize);
			for (int i = 0; i < curbatch; i++)
				d_CTFSimulate(v_params[b + i], d_ctfpoints, d_batchsim + denselength * i, denselength, true);
			d_NormMonolithic(d_batchsim, d_batchsim, denselength, (tfloat*)NULL, T_NORM_MEAN01STD, curbatch);
			d_MultiplyByVector(d_batchsim, d_ps, d_batchsim, denselength, curbatch);
			d_SumMonolithic(d_batchsim, d_batchscores + b, denselength, curbatch);
		}
		cudaMemcpy(h_batchscores, d_batchscores, v_params.size() * sizeof(tfloat), cudaMemcpyDeviceToHost);
		for (int i = 0; i < v_params.size(); i++)
		{
			h_batchscores[i] /= (tfloat)denselength;
			scores.push_back(h_batchscores[i]);
			if (h_batchscores[i] > bestscore)
			{
				bestscore = h_batchscores[i];
				bestfit = v_params[i];
			}
		}
		free(h_batchscores);
		cudaFree(d_batchscores);
		cudaFree(d_batchsim);

		tfloat3* h_p = (tfloat3*)&p;
		tfloat* h_bestfit = (tfloat*)&bestfit;
		for (int j = 0; j < 11; j++)
			if (h_p[j].x != h_p[j].y)
				h_p[j] = tfloat3(h_bestfit[j] - h_p[j].z * 1.5f, h_bestfit[j] + h_p[j].z * 1.5f, h_p[j].z / 2.0f);
	}

	cudaFree(d_simulated);
	cudaFree(d_ctfpoints);
	cudaFree(d_ps);
	
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