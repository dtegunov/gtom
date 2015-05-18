#include "Prerequisites.cuh"

#ifndef PCA_CUH
#define PCA_CUH

namespace gtom
{
	//Filter.cu:
	void d_PCAFilter(tfloat* d_input, int length, int samples, int ncomponents, tfloat* d_filtered);
	void d_PCAReconstruct(tfloat* d_eigenvectors, tfloat* d_eigenvalues, int length, int samples, int ncomponents, tfloat* d_output);

	//NIPALS.cu:

	/**
	* \brief Performs Princple Component Analysis using the NIPALS algorithm – less precise than GS, but slightly faster.
	* \param[in] d_data	Array in device memory with the original data
	* \param[in] samples	Number of observation
	* \param[in] length	Length of a single observation
	* \param[in] ncomponents	The first n components to be computed
	* \param[in] d_eigenvalues	Array in device memory of size (samples * ncomponents) that will hold the eigenvalues (scores)
	* \param[in] d_eigenvectors	Array in device memory of size (length * ncomponents) that will hold the eigenvectors (coefficients)
	* \param[in] d_residual	Array in device memory of size (samples * ncomponents) that will hold the remaining variance not explained by n components
	* \param[in] maxiterations	Maximum number of iterations until convergence
	* \param[in] maxerror	Maximum error for convergence
	*/
	void d_PCANIPALS(tfloat* d_data, int samples, int length, int ncomponents, tfloat* d_eigenvalues, tfloat* d_eigenvectors, tfloat* d_residual, int maxiterations = 1000, tfloat maxerror = 1e-7);
	void d_PCA(tfloat* d_data, int samples, int length, int ncomponents, tfloat* d_eigenvalues, tfloat* d_eigenvectors);

	//GS.cu:

	/**
	* \brief Performs Princple Component Analysis using the GS algorithm – more precise than NIPALS, but slightly slower.
	* \param[in] d_data	Array in device memory with the original data
	* \param[in] samples	Number of observation
	* \param[in] length	Length of a single observation
	* \param[in] ncomponents	The first n components to be computed
	* \param[in] d_eigenvalues	Array in device memory of size (samples * ncomponents) that will hold the eigenvalues (scores)
	* \param[in] d_eigenvectors	Array in device memory of size (length * ncomponents) that will hold the eigenvectors (coefficients)
	* \param[in] d_residual	Array in device memory of size (samples * ncomponents) that will hold the remaining variance not explained by n components
	* \param[in] maxiterations	Maximum number of iterations until convergence
	* \param[in] maxerror	Maximum error for convergence
	*/
	void d_PCAGS(tfloat* d_data, int samples, int length, int ncomponents, tfloat* d_eigenvalues, tfloat* d_eigenvectors, tfloat* d_residual, int maxiterations = 1000, tfloat maxerror = 1e-7);
}
#endif