#include "Prerequisites.h"

TEST(PCA, NIPALS)
{
	cudaDeviceReset();

	//Case 1:
	{
		int samples = 10;
		int length = 6;
		int components = 4;

		tfloat* d_input = (tfloat*)CudaMallocFromBinaryFile("Data/PCA/Input_PCA.bin");
		tfloat* d_eigenvectors = CudaMallocValueFilled(length * components, (tfloat)0);
		tfloat* d_eigenvalues = CudaMallocValueFilled(samples * components, (tfloat)0);
		tfloat* d_residual = CudaMallocValueFilled(samples * length, (tfloat)0);

		d_PCANIPALS(d_input, samples, length, components, d_eigenvalues, d_eigenvectors, d_residual);

		tfloat* h_eigenvectors = (tfloat*)MallocFromDeviceArray(d_eigenvectors, length * components * sizeof(tfloat));
		tfloat* h_eigenvalues = (tfloat*)MallocFromDeviceArray(d_eigenvalues, samples * components * sizeof(tfloat));
		tfloat* h_residual = (tfloat*)MallocFromDeviceArray(d_residual, samples * length * sizeof(tfloat));

		free(h_eigenvectors);
		free(h_eigenvalues);
		free(h_residual);
	}

	cudaDeviceReset();
}