#include "Prerequisites.cuh"
#include "Generics.cuh"
#include "Helper.cuh"
#include "PCA.cuh"


__global__ void PCAReconstructKernel(tfloat* d_eigenvectors, tfloat* d_eigenvalues, uint length, uint samples, uint ncomponents, tfloat* d_output);


void d_PCAFilter(tfloat* d_input, int length, int samples, int ncomponents, tfloat* d_filtered)
{
	// Center the data
	tfloat* d_mean;
	cudaMalloc((void**)&d_mean, length * sizeof(tfloat));
	d_ReduceMean(d_input, d_mean, length, samples);

	// PCA
	tfloat* d_eigenvectors;
	cudaMalloc((void**)&d_eigenvectors, length * ncomponents * sizeof(tfloat));
	tfloat* d_eigenvalues;
	cudaMalloc((void**)&d_eigenvalues, samples * ncomponents * sizeof(tfloat));
	tfloat* d_residuals;
	cudaMalloc((void**)&d_residuals, samples * length * sizeof(tfloat));

	d_PCANIPALS(d_input, samples, length, ncomponents, d_eigenvalues, d_eigenvectors, d_residuals);

	// Reconstruct and add previously subtracted mean
	d_PCAReconstruct(d_eigenvectors, d_eigenvalues, length, samples, ncomponents, d_filtered);
	d_AddVector(d_filtered, d_mean, d_filtered, length, samples);

	// Clean up
	cudaFree(d_residuals);
	cudaFree(d_eigenvalues);
	cudaFree(d_eigenvectors);
	cudaFree(d_mean);
}


void d_PCAReconstruct(tfloat* d_eigenvectors, tfloat* d_eigenvalues, int length, int samples, int ncomponents, tfloat* d_output)
{
	dim3 TpB = dim3(min(192, NextMultipleOf(length, 32)));
	dim3 grid = dim3(min(32768, samples));
	PCAReconstructKernel <<<grid, TpB>>> (d_eigenvectors, d_eigenvalues, length, samples, ncomponents, d_output);
}


__global__ void PCAReconstructKernel(tfloat* d_eigenvectors, tfloat* d_eigenvalues, uint length, uint samples, uint ncomponents, tfloat* d_output)
{
	for (uint sample = blockIdx.x; sample < samples; sample += gridDim.x)
	{
		for (uint element = threadIdx.x; element < length; element += blockDim.x)
		{
			tfloat sum = 0;
			for (uint component = 0; component < ncomponents; component++)
			{
				tfloat vectorelement = d_eigenvectors[component * length + element];
				tfloat value = d_eigenvalues[sample * ncomponents + component];
				sum += vectorelement * value;
			}
			d_output[sample * length + element] = sum;
		}
	}
}