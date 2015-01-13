#include "Prerequisites.cuh"
#include "Generics.cuh"
#include "Helper.cuh"
#include "Optimization.cuh"

using namespace dlib;

class d_poly_model
{

public:
	typedef ::column_vector column_vector;
	typedef matrix<double> general_matrix;

	tfloat* d_x;
	tfloat* d_y;
	tfloat* d_prepow;
	tfloat* d_sim;
	tfloat* d_r;
	general_matrix prehess;

	int npoints;

	d_poly_model(tfloat* _d_x, tfloat* _d_y, int _npoints, int degree)
	{
		d_x = _d_x;
		d_y = _d_y;
		npoints = _npoints;
		cudaMalloc((void**)&d_r, npoints * sizeof(tfloat));
		cudaMalloc((void**)&d_sim, npoints * sizeof(tfloat));
		
		cudaMalloc((void**)&d_prepow, npoints * degree * sizeof(tfloat));
		d_ValueFill(d_prepow, npoints, (tfloat)1);
		cudaMemcpy(d_prepow + npoints, d_x, npoints * sizeof(tfloat), cudaMemcpyDeviceToDevice);
		for (int i = 2; i < degree; i++)
			d_MultiplyByVector(d_prepow + npoints * (i - 1), d_x, d_prepow + npoints * i, npoints);

		tfloat* d_hess, *d_temp, *d_temp2;
		cudaMalloc((void**)&d_hess, degree * degree * sizeof(tfloat));
		cudaMalloc((void**)&d_temp, npoints * degree * sizeof(tfloat));
		cudaMalloc((void**)&d_temp2, degree * sizeof(tfloat));
		tfloat* h_temp2 = (tfloat*)malloc(degree * sizeof(tfloat));
		prehess = general_matrix(degree, degree);
		
		for (int i = 0; i < degree; i++)
		{
			d_MultiplyByVector(d_prepow, d_prepow + npoints * i, d_temp, npoints, degree);
			d_SumMonolithic(d_temp, d_temp2, npoints, degree);
			cudaMemcpy(h_temp2, d_temp2, degree * sizeof(tfloat), cudaMemcpyDeviceToHost);
			for (int j = 0; j < degree; j++)
				prehess(i, j) = h_temp2[j];
		}

		free(h_temp2);
		cudaFree(d_hess);
		cudaFree(d_temp);
		cudaFree(d_temp2);
	}

	~d_poly_model()
	{
		cudaFree(d_r);
		cudaFree(d_sim);
		cudaFree(d_prepow);
	}

	double operator() (const column_vector& params) const 
	{
		tfloat sum = 0;
		tfloat* h_params = (tfloat*)malloc(params.size() * sizeof(tfloat));
		for (int i = 0; i < params.size(); i++)
			h_params[i] = params(i);
		tfloat* d_params = (tfloat*)CudaMallocFromHostArray(h_params, params.size() * sizeof(tfloat));
		free(h_params);

		d_Polynomial1D(d_x, d_sim, npoints, d_params, params.size(), 1);
		cudaFree(d_params);

		d_SubtractVector(d_sim, d_y, d_r, npoints);
		d_MultiplyByVector(d_r, d_r, d_sim, npoints);
		d_Sum(d_sim, d_sim, npoints);
		cudaMemcpy(&sum, d_sim, sizeof(tfloat), cudaMemcpyDeviceToHost);

		return (tfloat)0.5 * sum;
	}

	void get_derivative_and_hessian(const column_vector& x, column_vector& der, general_matrix& hess) const
	{
		der = column_vector(x.size());
		for (int i = 0; i < x.size(); i++)
		{
			tfloat sum = 0;
			d_MultiplyByVector(d_prepow + npoints * i, d_r, d_sim, npoints);
			d_Sum(d_sim, d_sim, npoints);
			cudaMemcpy(&sum, d_sim, sizeof(tfloat), cudaMemcpyDeviceToHost);
			der(i) = sum;
		}
		
		hess = prehess;
	}
};

class h_poly_model
{

public:
	typedef ::column_vector column_vector;
	typedef matrix<double> general_matrix;

	tfloat* h_y;
	general_matrix prepow;
	mutable column_vector r;
	general_matrix prehess;

	int npoints, degree;

	h_poly_model(tfloat* _h_x, tfloat* _h_y, int _npoints, int _degree)
	{
		npoints = _npoints;
		degree = _degree;
		h_y = _h_y;
		prepow = general_matrix(npoints, degree);
		r = column_vector(npoints);
		for (int i = 0; i < npoints; i++)
		{
			prepow(i, 0) = 1.0;
			double x = _h_x[i];
			prepow(i, 1) = x;
			for (int f = 2; f < degree; f++)
				prepow(i, f) = prepow(i, f - 1)*x;
		}

		prepow = trans(prepow);
		prehess = prepow * trans(prepow);
	}

	double operator() (const column_vector& params) const
	{
		double sum = 0;
		for (int i = 0; i < npoints; i++)
		{
			double val = params(0);
			for (int f = 1; f < degree; f++)
				val += params(f) * prepow(f, i);
			val -= h_y[i];
			sum += val * val;
			r(i) = val;
		}

		return 0.5 * sum;
	}

	void get_derivative_and_hessian(const column_vector& x, column_vector& der, general_matrix& hess) const
	{
		der = prepow * r;
		hess = prehess;
	}
};

void d_PolynomialFit(tfloat* d_x, tfloat* d_y, int npoints, tfloat* d_factors, int degree)
{
	
	column_vector p = column_vector(degree);
	for (int f = 0; f < degree; f++)
		p(f) = 0.0;

	find_min_trust_region(objective_delta_stop_strategy(1e-4),
						  d_poly_model(d_x, d_y, npoints, degree),
						  p);

	tfloat* h_factors = (tfloat*)malloc(degree * sizeof(tfloat));
	for (int f = 0; f < degree; f++)
		h_factors[f] = p(f);
	cudaMemcpy(d_factors, h_factors, degree * sizeof(tfloat), cudaMemcpyHostToDevice);
}

void h_PolynomialFit(tfloat* h_x, tfloat* h_y, int npoints, tfloat* h_factors, int degree)
{
	column_vector p = column_vector(degree);
	for (int f = 0; f < degree; f++)
		p(f) = 0.0;

	find_min_trust_region(objective_delta_stop_strategy(1e-4),
						  h_poly_model(h_x, h_y, npoints, degree),
						  p);

	for (int f = 0; f < degree; f++)
		h_factors[f] = p(f);
}