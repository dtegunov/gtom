#pragma once
#include "cufft.h"
#include "Prerequisites.cuh"


/////////////////////
//Fourier transform//
/////////////////////

//FFT.cu:

/**
* \brief Performs forward FFT on real-valued data; output will have dimensions.x / 2 + 1 as its width
* \param[in] d_input	Array with input data
* \param[in] d_output	Array that will contain transformed data
* \param[in] ndimensions	Number of dimensions
* \param[in] dimensions	Array dimensions
* \param[in] batch	Number of arrays to transform
*/
void d_FFTR2C(tfloat* const d_input, tcomplex* const d_output, int const ndimensions, int3 const dimensions, int batch = 1);

/**
* \brief Performs forward FFT on real-valued data using a pre-cooked plan; output will have dimensions.x / 2 + 1 as its width
* \param[in] d_input	Array with input data
* \param[in] d_output	Array that will contain transformed data
* \param[in] plan	Pre-cooked plan for forward transform; can be obtained with d_FFTR2CGetPlan()
*/
void d_FFTR2C(tfloat* const d_input, tcomplex* const d_output, cufftHandle* plan);

/**
* \brief Performs forward FFT on real-valued data; output will have the same dimensions as input and contain the symmetrically redundant half
* \param[in] d_input	Array with input data
* \param[in] d_output	Array that will contain transformed data
* \param[in] ndimensions	Number of dimensions
* \param[in] dimensions	Array dimensions
* \param[in] batch	Number of arrays to transform
*/
void d_FFTR2CFull(tfloat* const d_input, tcomplex* const d_output, int const ndimensions, int3 const dimensions, int batch = 1);

/**
* \brief Performs forward FFT on complex-valued data
* \param[in] d_input	Array with input data
* \param[in] d_output	Array that will contain transformed data
* \param[in] ndimensions	Number of dimensions
* \param[in] dimensions	Array dimensions
* \param[in] batch	Number of arrays to transform
*/
void d_FFTC2C(tcomplex* const d_input, tcomplex* const d_output, int const ndimensions, int3 const dimensions, int batch = 1);

/**
* \brief Performs forward FFT on real-valued data contained in a host array; output will have dimensions.x / 2 + 1 as its width
* \param[in] h_input	Host array with input data
* \param[in] h_output	Host array that will contain transformed data
* \param[in] ndimensions	Number of dimensions
* \param[in] dimensions	Array dimensions
* \param[in] batch	Number of arrays to transform
*/
void FFTR2C(tfloat* const h_input, tcomplex* const h_output, int const ndimensions, int3 const dimensions, int batch = 1);

/**
* \brief Performs forward FFT on real-valued data contained in a host array; output will have the same dimensions as input and contain the symmetrically redundant half
* \param[in] h_input	Host array with input data
* \param[in] h_output	Host array that will contain transformed data
* \param[in] ndimensions	Number of dimensions
* \param[in] dimensions	Array dimensions
* \param[in] batch	Number of arrays to transform
*/
void FFTR2CFull(tfloat* const h_input, tcomplex* const h_output, int const ndimensions, int3 const dimensions, int batch = 1);

/**
* \brief Performs forward FFT on complex-valued data contained in a host array
* \param[in] h_input	Host array with input data
* \param[in] h_output	Host array that will contain transformed data
* \param[in] ndimensions	Number of dimensions
* \param[in] dimensions	Array dimensions
* \param[in] batch	Number of arrays to transform
*/
void FFTC2C(tcomplex* const h_input, tcomplex* const h_output, int const ndimensions, int3 const dimensions, int batch = 1);


/**
* \brief Pre-cooks a plan for forward FFT on real-valued data; must be used with same parameters as d_FFTR2C later
* \param[in] ndimensions	Number of dimensions
* \param[in] dimensions	Array dimensions
* \param[in] batch	Number of arrays to transform
*/
cufftHandle d_FFTR2CGetPlan(int const ndimensions, int3 const dimensions, int batch = 1);

//IFFT.cu:

/**
* \brief Performs inverse FFT on complex-valued data; input must have dimensions.x / 2 + 1 as its width
* \param[in] d_input	Array with input data
* \param[in] d_output	Array that will contain transformed data
* \param[in] ndimensions	Number of dimensions
* \param[in] dimensions	Array dimensions
* \param[in] batch	Number of arrays to transform
*/
void d_IFFTC2R(tcomplex* const d_input, tfloat* const d_output, int const ndimensions, int3 const dimensions, int batch = 1);

/**
* \brief Performs inverse FFT on complex-valued data using a pre-cooked plan; input must have dimensions.x / 2 + 1 as its width
* \param[in] d_input	Array with input data
* \param[in] d_output	Array that will contain transformed data
* \param[in] plan	Pre-cooked inverse FFT plan; can be obtained with d_IFFTC2RGetPlan()
* \param[in] dimensions	Array dimensions
* \param[in] batch	Number of arrays to transform
*/
void d_IFFTC2R(tcomplex* const d_input, tfloat* const d_output, cufftHandle* plan, int3 const dimensions, int batch = 1);

/**
* \brief Performs inverse FFT on complex-valued data using a pre-cooked plan; input must have dimensions.x / 2 + 1 as its width
* \param[in] d_input	Array with input data
* \param[in] d_output	Array that will contain transformed data
* \param[in] plan	Pre-cooked inverse FFT plan; can be obtained with d_IFFTC2RGetPlan()
*/
void d_IFFTC2R(tcomplex* const d_input, tfloat* const d_output, cufftHandle* plan);

/**
* \brief Performs inverse FFT on complex-valued data without Hermitian symmetry (thus width = dimensions.x) and discards the complex part of the output
* \param[in] d_input	Array with input data
* \param[in] d_output	Array that will contain transformed data
* \param[in] ndimensions	Number of dimensions
* \param[in] dimensions	Array dimensions
* \param[in] batch	Number of arrays to transform
*/
void d_IFFTC2RFull(tcomplex* const d_input, tfloat* const d_output, int const ndimensions, int3 const dimensions, int batch = 1);

/**
* \brief Performs inverse FFT on complex-valued data without Hermitian symmetry (thus width = dimensions.x)
* \param[in] d_input	Array with input data
* \param[in] d_output	Array that will contain transformed data
* \param[in] ndimensions	Number of dimensions
* \param[in] dimensions	Array dimensions
* \param[in] batch	Number of arrays to transform
*/
void d_IFFTC2C(tcomplex* const d_input, tcomplex* const d_output, int const ndimensions, int3 const dimensions, int batch = 1);

/**
* \brief Performs inverse FFT on complex-valued data without Hermitian symmetry (thus width = dimensions.x) using a pre-cooked plan
* \param[in] d_input	Array with input data
* \param[in] d_output	Array that will contain transformed data
* \param[in] dimensions	Array dimensions
*/
void d_IFFTC2C(tcomplex* const d_input, tcomplex* const d_output, cufftHandle* plan, int3 const dimensions);

/**
* \brief Performs inverse FFT on complex-valued data in a host array; input must have dimensions.x / 2 + 1 as its width
* \param[in] h_input	Array with input data
* \param[in] h_output	Array that will contain transformed data
* \param[in] ndimensions	Number of dimensions
* \param[in] dimensions	Array dimensions
* \param[in] batch	Number of arrays to transform
*/
void IFFTC2R(tcomplex* const h_input, tfloat* const h_output, int const ndimensions, int3 const dimensions, int batch = 1);

/**
* \brief Performs inverse FFT on complex-valued data without Hermitian symmetry (thus width = dimensions.x) in a host array and discards the complex part of the output
* \param[in] h_input	Array with input data
* \param[in] h_output	Array that will contain transformed data
* \param[in] ndimensions	Number of dimensions
* \param[in] dimensions	Array dimensions
* \param[in] batch	Number of arrays to transform
*/
void IFFTC2RFull(tcomplex* const h_input, tfloat* const h_output, int const ndimensions, int3 const dimensions, int batch = 1);

/**
* \brief Performs inverse FFT on complex-valued data without Hermitian symmetry (thus width = dimensions.x) in a host array
* \param[in] h_input	Array with input data
* \param[in] h_output	Array that will contain transformed data
* \param[in] ndimensions	Number of dimensions
* \param[in] dimensions	Array dimensions
* \param[in] batch	Number of arrays to transform
*/
void IFFTC2C(tcomplex* const h_input, tcomplex* const h_output, int const ndimensions, int3 const dimensions, int batch = 1);

/**
* \brief Performs inverse FFT on double-precision, real-valued data
* \param[in] d_input	Array with input data
* \param[in] d_output	Array that will contain transformed data
* \param[in] ndimensions	Number of dimensions
* \param[in] dimensions	Array dimensions
* \param[in] batch	Number of arrays to transform
*/
void d_IFFTZ2D(cufftDoubleComplex* const d_input, double* const d_output, int const ndimensions, int3 const dimensions, int batch = 1);


/**
* \brief Pre-cooks a plan for inverse FFT on complex-valued data with Hermitian symmetry (thus width = dimensions.x / 2 + 1); must be used with same parameters as d_IFFTC2R later
* \param[in] ndimensions	Number of dimensions
* \param[in] dimensions	Array dimensions
* \param[in] batch	Number of arrays to transform
*/
cufftHandle d_IFFTC2RGetPlan(int const ndimensions, int3 const dimensions, int batch = 1);

/**
* \brief Pre-cooks a plan for inverse FFT on complex-valued data without Hermitian symmetry (thus width = dimensions.x); must be used with same parameters as d_IFFTC2C later
* \param[in] ndimensions	Number of dimensions
* \param[in] dimensions	Array dimensions
* \param[in] batch	Number of arrays to transform
*/
cufftHandle d_IFFTC2CGetPlan(int const ndimensions, int3 const dimensions, int batch = 1);

//HermitianSymmetry.cu:

/**
* \brief Adds Hermitian-symmetric half to complex-valued data
* \param[in] d_input	Array with input data
* \param[in] d_output	Array that will contain padded data
* \param[in] dims	Array dimensions with symmetric half, i. e. input will have width = dimensions.x / 2 + 1
* \param[in] batch	Number of arrays to pad
*/
void d_HermitianSymmetryPad(tcomplex* const d_input, tcomplex* const d_output, int3 const dims, int batch = 1);

/**
* \brief Removes Hermitian-symmetric half from complex-valued data
* \param[in] d_input	Array with input data
* \param[in] d_output	Array that will contain trimmed data
* \param[in] dims	Array dimensions with symmetric half, i. e. output will have width = dimensions.x / 2 + 1
* \param[in] batch	Number of arrays to trim
*/
void d_HermitianSymmetryTrim(tcomplex* const d_input, tcomplex* const d_output, int3 const dims, int batch = 1);

/**
* \brief Ensures Hermitian symmetry in complex-valued data
* \param[in] d_input	Array with input data
* \param[in] d_output	Array that will contain symmetric data
* \param[in] dims	Array dimensions
* \param[in] batch	Number of arrays to symmetrize
*/
void d_HermitianSymmetryMirrorHalf(tcomplex* d_input, tcomplex* d_output, int3 dims, int batch = 1);

//FFTRemap.cu:
template <class T> void d_RemapFull2HalfFFT(T* d_input, T* d_output, int3 dims, int batch = 1);
template <class T> void d_RemapFullFFT2Full(T* d_input, T* d_output, int3 dims, int batch = 1);
template <class T> void d_RemapFull2FullFFT(T* d_input, T* d_output, int3 dims, int batch = 1);
template <class T> void d_RemapHalfFFT2Half(T* d_input, T* d_output, int3 dims, int batch = 1);
template <class T> void d_RemapHalf2HalfFFT(T* d_input, T* d_output, int3 dims, int batch = 1);

//FFTResize.cu:
template <class T> void d_FFTCrop(T* d_input, T* d_output, int3 olddims, int3 newdims, int batch = 1);
template <class T> void d_FFTFullCrop(T* d_input, T* d_output, int3 olddims, int3 newdims, int batch = 1);
template <class T> void d_FFTPad(T* d_input, T* d_output, int3 olddims, int3 newdims, int batch = 1);
template <class T> void d_FFTFullPad(T* d_input, T* d_output, int3 olddims, int3 newdims, int batch = 1);