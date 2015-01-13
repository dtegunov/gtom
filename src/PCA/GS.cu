#include "Prerequisites.cuh"
#include "Helper.cuh"

// C/C++ example for the CUBLAS (NVIDIA)
// implementation of GS-PCA algorithm
//
// M. Andrecut (c) 2008
//
// Published in "Parallel GPU Implementation of Iterative PCA Algorithms"
// Adopted for CUBLAS 2

void d_PCAGS(tfloat* d_data, int samples, int length, int ncomponents, tfloat* d_eigenvalues, tfloat* d_eigenvectors, tfloat* d_residual, int maxiterations, tfloat maxerror)
{
	// PCA model: X = TP’ + R
	// input: X, MxN matrix (data)
	// input: M = number of rows in X
	// input: N = number of columns in X
	// input: K = number of components (K<=N)
	// output: T, MxK scores matrix
	// output: P, NxK loads matrix
	// output: R, MxN residual matrix

	cublasHandle_t handle;
	cublasCreate(&handle);

	// Allocate memory for eigenvalues
	tfloat* h_L;
	h_L = (tfloat*)malloc(ncomponents * sizeof(tfloat));;

	// Transpose input
	tfloat one = 1.0, zero = 0, minone = -1.0;
	cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, samples, length, &one, d_data, length, &zero, d_data, length, d_residual, samples);

	// Mean center the data
	tfloat* d_U;
	cudaMalloc((void**)&d_U, samples * length * sizeof(tfloat));
	tfloat alpha = 1.0f;
	cublasScopy(handle, samples, d_residual, 1, d_U, 1);
	for (int n = 1; n < length; n++)
		cublasSaxpy(handle, samples, &alpha, d_residual + samples * n, 1, d_U, 1);

	alpha = -1.0f / (tfloat)length;
	for (int n = 0; n < length; n++)
		cublasSaxpy(handle, samples, &alpha, d_U, 1, d_residual + samples * n, 1);
	
	// GS-PCA
	tfloat a;
	tfloat Snrm2Result = 0;
	for (int k = 0; k < ncomponents; k++)
	{
		cublasScopy(handle, samples, d_residual + samples * k, 1, d_eigenvalues + samples * k, 1);		a = 0.0;
		for (int j = 0; j < maxiterations; j++)
		{
			cublasSgemv(handle, CUBLAS_OP_T, samples, length, &one, d_residual, samples, d_eigenvalues + samples * k, 1, &zero, d_eigenvectors + length * k, 1);
			if (k > 0)
			{
				cublasSgemv(handle, CUBLAS_OP_T, length, k, &one, d_eigenvectors, length, d_eigenvectors + length * k, 1, &zero, d_U, 1);
				cublasSgemv(handle, CUBLAS_OP_N, length, k, &minone, d_eigenvectors, length, d_U, 1, &one, d_eigenvectors + length * k, 1);
			}
			cublasSnrm2(handle, length, d_eigenvectors + length * k, 1, &Snrm2Result);
			{
				tfloat alpha = 1.0 / Snrm2Result;
				cublasSscal(handle, length, &alpha, d_eigenvectors + length * k, 1);
			}
			cublasSgemv(handle, CUBLAS_OP_N, samples, length, &one, d_residual, samples, d_eigenvectors + length * k, 1, &zero, d_eigenvalues + samples * k, 1);
			if (k > 0)
			{
				cublasSgemv(handle, CUBLAS_OP_T, samples, k, &one, d_eigenvalues, samples, d_eigenvalues + samples * k, 1, &zero, d_U, 1);
				cublasSgemv(handle, CUBLAS_OP_N, samples, k, &minone, d_eigenvalues, samples, d_U, 1, &one, d_eigenvalues + samples * k, 1);
			}
			cublasSnrm2(handle, samples, d_eigenvalues + samples * k, 1, h_L + k);
			{
				tfloat alpha = 1.0 / h_L[k];
				cublasSscal(handle, samples, &alpha, d_eigenvalues + samples * k, 1);
			}
			if (fabs(a - h_L[k]) < maxerror * h_L[k]) 
				break;
			a = h_L[k];
		}
		{
			tfloat alpha = -h_L[k];
			cublasSger(handle, samples, length, &alpha, d_eigenvalues + samples * k, 1, d_eigenvectors + length * k, 1, d_residual, samples);
		}
	}
	for (int k = 0; k < ncomponents; k++)
		cublasSscal(handle, samples, h_L + k, d_eigenvalues + samples * k, 1);

	// Transpose eigenvalues
	cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, ncomponents, samples, &one, d_eigenvalues, samples, &zero, d_eigenvalues, samples, d_U, ncomponents);
	cudaMemcpy(d_eigenvalues, d_U, samples * ncomponents * sizeof(tfloat), cudaMemcpyDeviceToDevice);

	// Transpose residual
	cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, length, samples, &one, d_residual, samples, &zero, d_residual, samples, d_U, length);
	cudaMemcpy(d_residual, d_U, samples * length * sizeof(tfloat), cudaMemcpyDeviceToDevice);

	free(h_L);
	cudaFree(d_U);
}