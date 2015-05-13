#pragma once
#include "Prerequisites.cuh"
#include "../src/Optimization/dlib/optimization.h"
#include "../src/Optimization/polynomials/polynomial.h"

typedef dlib::matrix<double, 0, 1> column_vector;


enum T_OPTIMIZATION_MODE
{
	T_OPTIMIZATION_BRUTE = 1 << 0,
	T_OPTIMIZATION_ITERATIVE = 1 << 1,
	T_OPTIMIZATION_NEWTON = 1 << 2
};


////////////////
//Optimization//
////////////////

class Optimizer
{
public:
	double psi;

	virtual double Evaluate(column_vector& vec) = 0;

	virtual column_vector Derivative(column_vector x)
	{
		column_vector d = column_vector(x.size());
		for (int i = 0; i < x.size(); i++)
			d(i) = x(i);

		for (int i = 0; i < x.size(); i++)
		{
			double old = x(i);

			x(i) = old + psi;
			double dp = this->Evaluate(x);
			x(i) = old - psi;
			double dn = this->Evaluate(x);

			x(i) = old;
			d(i) = (dp - dn) / (2.0 * psi);
		}

		return d;
	}
};

class EvalWrapper
{
public:

	EvalWrapper(Optimizer* input)
	{
		wrapped = input;
	}

	double operator() (column_vector const &arg) const
	{
		column_vector nonconstarg = column_vector(arg);
		return wrapped->Evaluate(nonconstarg);
	}

private:
	Optimizer* wrapped;
};

class DerivativeWrapper
{
public:

	DerivativeWrapper(Optimizer* input)
	{
		wrapped = input;
	}

	column_vector operator() (const column_vector& arg) const
	{
		return wrapped->Derivative((column_vector)arg);
	}

private:
	Optimizer* wrapped;
};

//OptimizeSPAParams.cu:
void d_OptimizeSPAParams(tfloat* d_images,
	tfloat* d_imagespsf,
	int2 dimsimage,
	int nimages,
	int indexhold,
	int maskradius,
	tfloat3* h_angles,
	tfloat2* h_shifts,
	tfloat2* h_intensities,
	tfloat3 deltaanglesmax,
	tfloat deltashiftmax,
	tfloat2 deltaintensitymax,
	tfloat &finalscore);

//OptimizeStackWarp2D.cu:
void d_OptimizeStackWarp2D(tfloat* d_images,
	int2 dimsimage,
	int2 dimsgrid,
	uint nimages,
	uint indexhold,
	int maxshift,
	tfloat2* h_grids,
	tfloat* h_scores);

//OptimizeTomoParams.cu:
void d_OptimizeTomoParams(tfloat* d_images,
	int2 dimsimage,
	int3 dimsvolume,
	int nimages,
	vector<int> indiceshold,
	tfloat3* h_angles,
	tfloat2* h_shifts,
	tfloat2* h_intensities,
	tfloat3 deltaanglesmax,
	tfloat deltashiftmax,
	tfloat2 deltaintensitymax,
	tfloat &finalscore);

//OptimizeTomoParamsWBP.cu:
void d_OptimizeTomoParamsWBP(tfloat* d_images,
	int2 dimsimage,
	int3 dimsvolume,
	int nimages,
	vector<int> &indiceshold,
	tfloat3* h_angles,
	tfloat2* h_shifts,
	tfloat3* h_deltaanglesmin, tfloat3* h_deltaanglesmax,
	tfloat2* h_deltashiftmin, tfloat2* h_deltashiftmax,
	tfloat &finalscore);

//PolynomialFit.cu:

void h_PolynomialFit(tfloat* h_x, tfloat* h_y, int npoints, tfloat* h_factors, int degree);
void d_PolynomialFit(tfloat* d_x, tfloat* d_y, int npoints, tfloat* h_factors, int degree);