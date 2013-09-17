#pragma once
#include "cufft.h"

//#define TOM_DOUBLE

#ifdef TOM_DOUBLE
	typedef double tfloat;
	typedef cufftDoubleComplex tcomplex;
	#define IS_TFLOAT_DOUBLE true
#else
	typedef float tfloat;
	typedef cufftComplex tcomplex;
	#define IS_TFLOAT_DOUBLE false
#endif


////////////////////
//Helper Functions//
////////////////////

//TypeConversion.cu:
template <class T> tfloat* ConvertToTFloat(T const* const original, size_t const n);
template <class T> void ConvertToTFloat(T const* const original, tfloat* const copy, size_t const n);
template <class T> T* ConvertTFloatTo(tfloat const* const original, size_t const n);
template <class T> void ConvertTFloatTo(tfloat const* const original, T* const copy, size_t const n);

template <class T> tcomplex* ConvertSplitComplexToTComplex(T const* const originalr, T const* const originali, size_t const n);
template <class T> void ConvertSplitComplexToTComplex(T const* const originalr, T const* const originali, tcomplex* const copy, size_t const n);
template <class T> T* ConvertTComplexToSplitComplex(tcomplex const* const original, size_t const n);
template <class T> void ConvertTComplexToSplitComplex(tcomplex const* const original, T* const copyr, T* const copyi, size_t const n);

template <class T> void d_ConvertToTFloat(T const* const d_original, tfloat* const d_copy, size_t const n);
template <class T> void d_ConvertTFloatTo(tfloat const* const d_original, T* const d_copy, size_t const n);
template <class T> void d_ConvertSplitComplexToTComplex(T const* const d_originalr, T const* const d_originali, tcomplex* const d_copy, size_t const n);
template <class T> void d_ConvertTComplexToSplitComplex(tcomplex const* const d_original, T* const d_copyr, T* const d_copyi, size_t const n);

//Memory.cu:

/**
 * \brief Allocates an array in host memory that is a copy of the array in device memory.
 * \param[in] d_array	Array in device memory to be copied
 * \param[in] size		Array size in bytes
 * \returns Array pointer in host memory
 */
void* MallocFromDeviceArray(void* d_array, size_t size);

/**
 * \brief Allocates an array in device memory that is a copy of the array in host memory.
 * \param[in] h_array	Array in host memory to be copied
 * \param[in] size		Array size in bytes
 * \returns Array pointer in device memory
 */
void* CudaMallocFromHostArray(void* h_array, size_t size);

/**
 * \brief Allocates an array in device memory that is a copy of the array in host memory, both arrays can have different size.
 * \param[in] h_array		Array in host memory to be copied
 * \param[in] devicesize	Array size in bytes
 * \param[in] hostsize		Portion of the host array to be copied in bytes
 * \returns Array pointer in device memory
 */
void* CudaMallocFromHostArray(void* h_array, size_t devicesize, size_t hostsize);

/**
 * \brief Creates an array of floats initialized to 0.0f in host memory with the specified element count.
 * \param[in] elements	Element count
 * \returns Array pointer in host memory
 */
tfloat* MallocZeroFilledFloat(size_t elements);

/**
 * \brief Creates an array of floats initialized to 0.0f in device memory with the specified element count.
 * \param[in] elements	Element count
 * \returns Array pointer in device memory
 */
float* CudaMallocZeroFilledFloat(size_t elements);


/////////////////////
//Array arithmetics//
/////////////////////

//Arithmetics.cu:
void d_Multiply(tfloat* d_input, tfloat* d_output, size_t elements, tfloat multiplicator);
void d_Add(tfloat* d_input, tfloat* d_output, size_t elements, tfloat summand);


/////////////////////
//Fourier transform//
/////////////////////

//FFT.cu:
void d_FFTR2C(tfloat* const d_input, tcomplex* const d_output, int const ndimensions, int3 const dimensions);
void d_FFTR2CFull(tfloat* const d_input, tcomplex* const d_output, int const ndimensions, int3 const dimensions);
void d_FFTC2C(tcomplex* const d_input, tcomplex* const d_output, int const ndimensions, int3 const dimensions);
void FFTR2C(tfloat* const h_input, tcomplex* const h_output, int const ndimensions, int3 const dimensions);
void FFTR2CFull(tfloat* const h_input, tcomplex* const h_output, int const ndimensions, int3 const dimensions);
void FFTC2C(tcomplex* const h_input, tcomplex* const h_output, int const ndimensions, int3 const dimensions);

//IFFT.cu:
void d_IFFTC2R(tcomplex* const d_input, tfloat* const d_output, int const ndimensions, int3 const dimensions);
void d_IFFTC2RFull(tcomplex* const d_input, tfloat* const d_output, int const ndimensions, int3 const dimensions);
void d_IFFTC2C(tcomplex* const d_input, tcomplex* const d_output, int const ndimensions, int3 const dimensions);
void IFFTC2R(tcomplex* const h_input, tfloat* const h_output, int const ndimensions, int3 const dimensions);
void IFFTC2RFull(tcomplex* const h_input, tfloat* const h_output, int const ndimensions, int3 const dimensions);
void IFFTC2C(tcomplex* const h_input, tcomplex* const h_output, int const ndimensions, int3 const dimensions);

//HermitianSymmetry.cu:
void d_HermitianSymmetryPad(tcomplex* const d_input, tcomplex* const d_output, int3 const dimensions);
void d_HermitianSymmetryTrim(tcomplex* const d_input, tcomplex* const d_output, int3 const dimensions);

