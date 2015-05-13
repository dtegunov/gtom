#include "Prerequisites.cuh"
#include "Generics.cuh"
#include "Helper.cuh"

#include "cusolverDn.h"

// C/C++ example for the CUBLAS (NVIDIA)
// implementation of NIPALS-PCA algorithm
//
// M. Andrecut (c) 2008
//
// Published in "Parallel GPU Implementation of Iterative PCA Algorithms"
// Adopted for CUBLAS 2

void d_PCANIPALS(tfloat* d_data, int samples, int length, int ncomponents, tfloat* d_eigenvalues, tfloat* d_eigenvectors, tfloat* d_residual, int maxiterations, tfloat maxerror)
{
	cublasHandle_t handle;
	cublasCreate(&handle);

	tfloat* d_U;
	cudaMalloc((void**)&d_U, samples * length * sizeof(tfloat));
	tfloat alpha = 1.0f;

	// Mean center the data
	tfloat* d_mean;
	cudaMalloc((void**)&d_mean, length * sizeof(tfloat));
	d_ReduceMean(d_data, d_mean, length, samples);
	d_SubtractVector(d_data, d_mean, d_U, length, samples);	
	cudaFree(d_mean);

	// Transpose the centered data
	tfloat one = 1.0, zero = 0;
	cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, samples, length, &one, d_U, length, &zero, d_U, length, d_residual, samples);

	tfloat a, b;
	tfloat Snrm2Result = 0;
	for (int k = 0; k < ncomponents; k++)
	{
		cublasScopy(handle, samples, d_residual + samples * k, 1, d_eigenvalues + samples * k, 1);
		a = 0.0;
		for (int j = 0; j < maxiterations; j++)
		{
			alpha = 1.0f;
			tfloat beta = 0;
			cublasSgemv(handle, CUBLAS_OP_T, samples, length, &alpha, d_residual, samples, d_eigenvalues + samples * k, 1, &beta, d_eigenvectors + length * k, 1);
			cublasSnrm2(handle, length, d_eigenvectors + length * k, 1, &Snrm2Result);
			Snrm2Result = 1.0 / Snrm2Result;
			cublasSscal(handle, length, &Snrm2Result, d_eigenvectors + length * k, 1);
			cublasSgemv(handle, CUBLAS_OP_N, samples, length, &alpha, d_residual, samples, d_eigenvectors + length * k, 1, &beta, d_eigenvalues + samples * k, 1);
			cublasSnrm2(handle, samples, d_eigenvalues + samples * k, 1, &b);
			if (abs(a - b) < maxerror * b)
				break;
			a = b;
		}

		alpha = -1.0;
		cublasSger(handle, samples, length, &alpha, d_eigenvalues + samples * k, 1, d_eigenvectors + length * k, 1, d_residual, samples);
	}

	// Transpose eigenvalues
	cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, ncomponents, samples, &one, d_eigenvalues, samples, &zero, d_eigenvalues, samples, d_U, ncomponents);
	cudaMemcpy(d_eigenvalues, d_U, samples * ncomponents * sizeof(tfloat), cudaMemcpyDeviceToDevice);

	// Transpose residual
	cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, length, samples, &one, d_residual, samples, &zero, d_residual, samples, d_U, length);
	cudaMemcpy(d_residual, d_U, samples * length * sizeof(tfloat), cudaMemcpyDeviceToDevice);

	cudaFree(d_U);
	cublasDestroy(handle);
}