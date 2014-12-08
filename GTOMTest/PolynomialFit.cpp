#include "Prerequisites.h"

TEST(Optimization, PolynomialFit)
{
	cudaDeviceReset();

	//Case 1:
	{
		int npoints = 100;
		tfloat* h_x = (tfloat*)malloc(npoints * sizeof(tfloat));
		tfloat* h_y = (tfloat*)malloc(npoints * sizeof(tfloat));
		int degree = 4;
		tfloat* h_factor = (tfloat*)malloc(degree * sizeof(tfloat));
		h_factor[0] = 0.1234f;
		h_factor[1] = 0.123f;
		h_factor[2] = 0.12f;
		h_factor[3] = 0.1f;

		for (int n = 0; n < npoints; n++)
		{
			h_x[n] = n;
			h_y[n] = pow(h_x[n], 0.0) * h_factor[0] + pow(h_x[n], 1.0) * h_factor[1] + pow(h_x[n], 2.0) * h_factor[2] + pow(h_x[n], 3.0) * h_factor[3];
		}

		tfloat* h_factors = (tfloat*)malloc(degree * sizeof(tfloat));

		tfloat* d_x = (tfloat*)CudaMallocFromHostArray(h_x, npoints * sizeof(tfloat));
		tfloat* d_y = (tfloat*)CudaMallocFromHostArray(h_y, npoints * sizeof(tfloat));
		tfloat* d_factors = CudaMallocValueFilled(degree, (tfloat)0);

		h_PolynomialFit(h_x, h_y, npoints, h_factors, degree);

		free(h_factors);
	}

	cudaDeviceReset();
}