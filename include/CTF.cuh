#pragma once
#include "Prerequisites.cuh"
#include "Optimization.cuh"

struct CTFParams
{
	tfloat pixelsize;
	tfloat Cs;
	tfloat Cc;
	tfloat voltage;
	tfloat defocus;
	tfloat astigmatismangle;
	tfloat defocusdelta;
	tfloat amplitude;
	tfloat Bfactor;
	tfloat decayCohIll;
	tfloat decayspread;

	CTFParams() :
		pixelsize(1e-10),
		Cs(2e-3),
		Cc(2.2e-3),
		voltage(300e3),
		defocus(-3e-6),
		astigmatismangle(0),
		defocusdelta(0),
		amplitude(0.07),
		Bfactor(0),
		decayCohIll(0.0),
		decayspread(0.8) {}
};

struct CTFFitParams
{
	tfloat3 pixelsize;
	tfloat3 Cs;
	tfloat3 Cc;
	tfloat3 voltage;
	tfloat3 defocus;
	tfloat3 astigmatismangle;
	tfloat3 defocusdelta;
	tfloat3 amplitude;
	tfloat3 Bfactor;
	tfloat3 decayCohIll;
	tfloat3 decayspread;

	int2 dimsperiodogram;
	int maskinnerradius;
	int maskouterradius;

	CTFFitParams() :
		pixelsize(1e-10),
		Cs(2e-3),
		Cc(2.2e-3),
		voltage(300e3),
		defocus(-3e-6),
		astigmatismangle(0),
		defocusdelta(0),
		amplitude(0.07),
		Bfactor(0),
		decayCohIll(0),
		decayspread(0.8),
		dimsperiodogram(toInt2(512, 512)),
		maskinnerradius(8),
		maskouterradius(128) {}

	CTFFitParams(CTFParams p) :
		pixelsize(p.pixelsize),
		Cs(p.Cs),
		Cc(p.Cc),
		voltage(p.voltage),
		defocus(p.defocus),
		astigmatismangle(p.astigmatismangle),
		defocusdelta(p.defocusdelta),
		amplitude(p.amplitude),
		Bfactor(p.Bfactor),
		decayCohIll(p.decayCohIll),
		decayspread(p.decayspread),
		dimsperiodogram(toInt2(512, 512)),
		maskinnerradius(8),
		maskouterradius(128) {}
};

struct CTFParamsLean
{
	double ny;
	double lambda;
	double lambda3;
	double Cs;
	double ccvoltage;
	double defocus;
	double astigmatismangle;
	double defocusdelta;
	double amplitude;
	double Bfactor;
	double decayCohIll;
	double decayspread;
	double q0;

	CTFParamsLean(CTFParams p) :
		ny(0.5f / p.pixelsize),
		lambda(sqrt(150.4 / (p.voltage * (1.0 + p.voltage / 1022000.0))) * 1e-10),
		lambda3(lambda * lambda * lambda),
		Cs(p.Cs),
		ccvoltage(p.Cc / p.voltage),
		defocus(p.defocus),
		astigmatismangle(p.astigmatismangle),
		defocusdelta(p.defocusdelta * 0.5),
		amplitude(p.amplitude),
		Bfactor(-p.Bfactor * 0.25),
		decayCohIll(p.decayCohIll),
		decayspread(p.decayspread),
		q0(p.decayCohIll / (pow((double)p.Cs * pow((double)lambda, 3.0), 0.25))){}
};

template<bool ampsquared> __device__ double d_GetCTF(double k, double angle, CTFParamsLean p)
{
	double k2 = k * k;
	double term1 = p.lambda3 * p.Cs * (k2 * k2);
	if (p.Bfactor == 0.0f)
		p.Bfactor = 1.0f;
	else
		p.Bfactor = exp(p.Bfactor * k2);

	double e_i_k = 1.0f;
	if (p.decayCohIll != 0.0f)
	{
		e_i_k = exp(-(PI * PI) * (p.q0 * p.q0) * pow((double)p.Cs * p.lambda3 * pow(k, 3.0) - p.defocus * p.lambda * k, 2.0));
	}

	double e_e_k = 1.0;
	if (p.decayspread != 0.0)
	{
		double delta_z = p.ccvoltage * p.decayspread;
		e_e_k = exp(-(PI * delta_z * p.lambda * 2.0 * k2));
	}

	double w = PIHALF * (term1 - p.lambda * 2.0 * (p.defocus + p.defocusdelta * cos(2.0 * (angle - p.astigmatismangle))) * k2);
	double amplitude = cos(w);
	double phase = sin(w);

	if (!ampsquared)
		amplitude = e_e_k * e_i_k * p.Bfactor * (sqrt(1 - p.amplitude * p.amplitude) * phase + p.amplitude * amplitude);
	else
	{
		amplitude = (sqrt(1 - p.amplitude * p.amplitude) * phase + p.amplitude * amplitude);
		amplitude = e_e_k * e_i_k * p.Bfactor * amplitude * amplitude;
	}

	return amplitude;
}


//Correct.cu:
void d_CTFCorrect(tcomplex* d_input, int3 dimsinput, CTFParams params, tcomplex* d_output);

//Decay.cu:

void d_CTFDecay(tfloat* d_input, tfloat* d_output, int2 dims, int degree, int stripwidth);

//Fit.cu:

void d_CTFFit(tfloat* d_image, int2 dimsimage, int3* d_origins, int norigins, CTFFitParams p, int refinements, CTFParams &fit, tfloat &score, tfloat &mean, tfloat &stddev);

//Periodogram.cu:

void d_Periodogram(tfloat* d_image, int2 dimsimage, int3* d_origins, int norigins, int2 dimsregion, tfloat* d_output);

//Simulate.cu:

void d_CTFSimulate(CTFParams params, float2* d_addresses, tfloat* d_output, uint n, bool amplitudesquared = false);

//Wiener.cu:

void d_WienerPerFreq(tcomplex* d_input, int3 dimsinput, tfloat* d_fsc, CTFParams params, tcomplex* d_output, tfloat* d_outputweights);