#include "Prerequisites.h"

TEST(PCA, Filter)
{
	cudaDeviceReset();

	//Case 1:
	{
		int samples = 10;
		int length = 6;
		int components = 4;

		tfloat* d_input = (tfloat*)CudaMallocFromBinaryFile("Data/PCA/Input_PCAFilter_data.bin");
		tfloat* d_filtered = CudaMallocValueFilled(length * samples, (tfloat)0);

		d_PCAFilter(d_input, length, samples, components, d_filtered);

		tfloat* d_eigenvectors = (tfloat*)CudaMallocFromBinaryFile("Data/PCA/Input_PCAFilter_eigenvecs.bin");
		tfloat* d_eigenvalues = (tfloat*)CudaMallocFromBinaryFile("Data/PCA/Input_PCAFilter_eigenvals.bin");
		d_PCAReconstruct(d_eigenvectors, d_eigenvalues, length, samples, components, d_filtered);
		//CudaWriteToBinaryFile("d_filtered.bin", d_filtered, length * samples * sizeof(tfloat));
	}

	cudaDeviceReset();
}