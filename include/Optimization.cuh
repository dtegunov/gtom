#pragma once
#include "Prerequisites.cuh"
#include "../src/Optimization/dlib/optimization.h"

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

	double operator() (const column_vector& arg) const
	{
		return wrapped->Evaluate((column_vector)arg);
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

//PolynomialFit.cu:

void h_PolynomialFit(tfloat* h_x, tfloat* h_y, int npoints, tfloat* h_factors, int degree);
void d_PolynomialFit(tfloat* d_x, tfloat* d_y, int npoints, tfloat* h_factors, int degree);