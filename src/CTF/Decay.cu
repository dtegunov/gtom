#include "Prerequisites.cuh"
#include "FFT.cuh"
#include "Generics.cuh"
#include "Helper.cuh"
#include "Optimization.cuh"
#include "Transformation.cuh"


namespace gtom
{
	//////////////////////////////////////////////////
	//Fit polynomial decay curve to a power spectrum//
	//////////////////////////////////////////////////

	void d_CTFDecay(tfloat* d_input, tfloat* d_output, int2 dims, int degree, int stripwidth)
	{
		int2 dimsstrip = toInt2(dims.x, stripwidth);
		tfloat* d_strips;
		cudaMalloc((void**)&d_strips, Elements2(dimsstrip) * dims.y * sizeof(tfloat));
		int2 dimspadded = toInt2(dims.x, dims.y + stripwidth);
		tfloat* d_inputpadded;
		cudaMalloc((void**)&d_inputpadded, Elements2(dimspadded) * sizeof(tfloat));

		d_Pad(d_input, d_inputpadded, toInt3(dims), toInt3(dimspadded), T_PAD_TILE, (tfloat)0);

		int3* h_positions = (int3*)malloc(dims.y * sizeof(int3));
		for (int y = 0; y < dims.y; y++)
			h_positions[y] = toInt3(0, y, 0);
		int3* d_positions = (int3*)CudaMallocFromHostArray(h_positions, dims.y * sizeof(int3));
		free(h_positions);
		d_ExtractMany(d_inputpadded, d_strips, toInt3(dimspadded), toInt3(dimsstrip), d_positions, dims.y);
		cudaFree(d_positions);
		cudaFree(d_inputpadded);

		d_ReduceMean(d_strips, d_strips, dims.x, stripwidth, dims.y);
		tfloat* h_sums = (tfloat*)MallocFromDeviceArray(d_strips, Elements2(dims) * sizeof(tfloat));
		cudaFree(d_strips);

		tfloat* h_x = (tfloat*)malloc(dims.x * sizeof(tfloat));
		for (int x = 0; x < dims.x; x++)
			h_x[x] = x;
		tfloat* h_factors = (tfloat*)malloc(degree * dims.y * sizeof(tfloat));

#pragma omp parallel for
		for (int y = 0; y < dims.y; y++)
			h_PolynomialFit(h_x, h_sums + dims.x * y, dims.x, h_factors + degree * y, degree);

		tfloat* d_x;
		cudaMalloc((void**)&d_x, Elements2(dims) * sizeof(tfloat));
		cudaMemcpy(d_x, h_x, dims.x * sizeof(tfloat), cudaMemcpyHostToDevice);
		CudaMemcpyMulti(d_x + dims.x, d_x, dims.x, dims.y - 1);
		tfloat* d_factors = (tfloat*)CudaMallocFromHostArray(h_factors, degree * dims.y * sizeof(tfloat));
		free(h_factors);
		free(h_x);

		d_Polynomial1D(d_x, d_output, dims.x, d_factors, degree, dims.y);

		cudaFree(d_factors);
		cudaFree(d_x);
	}
}