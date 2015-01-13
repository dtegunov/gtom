#pragma once
#include "Prerequisites.cuh"

//NIPALS.cu:
void d_PCANIPALS(tfloat* d_data, int samples, int length, int ncomponents, tfloat* d_eigenvalues, tfloat* d_eigenvectors, tfloat* d_residual, int maxiterations = 10000, tfloat maxerror = 1e-7);

//GS.cu:
void d_PCAGS(tfloat* d_data, int samples, int length, int ncomponents, tfloat* d_eigenvalues, tfloat* d_eigenvectors, tfloat* d_residual, int maxiterations = 10000, tfloat maxerror = 1e-7);