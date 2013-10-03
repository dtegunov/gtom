#pragma once
#include "cufft.h"
#include "Prerequisites.cuh"


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
 * \brief Creates an array of T initialized to value in host memory with the specified element count.
 * \param[in] elements	Element count
 * \param[in] value		Initial value
 * \returns Array pointer in host memory
 */
template <class T> T* MallocValueFilled(size_t elements, T value);

/**
 * \brief Creates an array of floats initialized to 0.0f in device memory with the specified element count.
 * \param[in] elements	Element count
 * \returns Array pointer in device memory
 */
float* CudaMallocZeroFilledFloat(size_t elements);

/**
 * \brief Creates an array of T initialized to value in device memory with the specified element count.
 * \param[in] elements	Element count
 * \param[in] value		Initial value
 * \returns Array pointer in device memory
 */
template <class T> T* CudaMallocValueFilled(size_t elements, T value);

/**
 * \brief Creates an array of T tuples with n fields and copies the scalar inputs to the corresponding fields, obtaining an interleaved memory layout.
 * \param[in] T				Data type
 * \param[in] fieldcount	Number of fields per tuple (and number of pointers in d_fields
 * \param[in] d_fields		Array with pointers to scalar input arrays of size = fieldcount
 * \param[in] elements		Number of elements per scalar input array
 * \returns Array pointer in device memory
 */
template <class T, int fieldcount> T* d_JoinInterleaved(T** d_fields, size_t elements);

/**
 * \brief Writes T tuples with n fields, filled with the scalar inputs, into the provided output array obtaining an interleaved memory layout.
 * \param[in] T				Data type
 * \param[in] fieldcount	Number of fields per tuple (and number of pointers in d_fields
 * \param[in] d_fields		Array with pointers to scalar input arrays of size = fieldcount
 * \param[in] d_output		Array of size = fieldcount * elements to which the tuples will be written
 * \param[in] elements		Number of elements per scalar input array
 */
template <class T, int fieldcount> void d_JoinInterleaved(T** d_fields, T* d_output, size_t elements);


///////////////
//Arithmetics//
///////////////

//Arithmetics.cu:
template <class T> void d_MultiplyByScalar(T* d_input, T* d_output, size_t elements, T multiplicator);
template <class T> void d_MultiplyByScalar(T* d_input, T* d_multiplicators, T* d_output, size_t elements, int batch = 1);
template <class T> void d_MultiplyByVector(T* d_input, T* d_multiplicators, T* d_output, size_t elements, int batch = 1);

void d_ComplexMultiplyByVector(tcomplex* d_input, tfloat* d_multiplicators, tcomplex* d_output, size_t elements, int batch = 1);
void d_ComplexMultiplyByScalar(tcomplex* d_input, tcomplex* d_output, size_t elements, tfloat multiplicator);
void d_ComplexMultiplyByScalar(tcomplex* d_input, tfloat* d_multiplicators, tcomplex* d_output, size_t elements, int batch = 1);

template <class T> void d_AddScalar(T* d_input, T* d_output, size_t elements, T summand);
template <class T> void d_AddScalar(T* d_input, T* d_summands, T* d_output, size_t elements, int batch = 1);
template <class T> void d_AddVector(T* d_input, T* d_summands, T* d_output, size_t elements, int batch = 1);

template <class T> void d_SubtractScalar(T* d_input, T* d_output, size_t elements, T subtrahend);
template <class T> void d_SubtractScalar(T* d_input, T* d_subtrahends, T* d_output, size_t elements, int batch = 1);
template <class T> void d_SubtractVector(T* d_input, T* d_subtrahends, T* d_output, size_t elements, int batch = 1);

template <class T> void d_Sqrt(T* d_input, T* d_output, size_t elements);
template <class T> void d_Square(T* d_input, T* d_output, size_t elements, int batch = 1);
template <class T> void d_Pow(T* d_input, T* d_output, size_t elements, T exponent);

size_t NextPow2(size_t x);
bool IsPow2(size_t x);

//CompositeArithmetics.cu
template <class T> void d_SquaredDistanceFromVector(T* d_input, T* d_vector, T* d_output, size_t elements, int batch = 1);
template <class T> void d_SquaredDistanceFromScalar(T* d_input, T* d_output, size_t elements, T scalar);
template <class T> void d_SquaredDistanceFromScalar(T* d_input, T* d_scalars, T* d_output, size_t elements, int batch = 1);

//Sum.cu:
template <class T> void d_Sum(T *d_input, T *d_output, size_t n, int batch = 1);

//MinMax.cu:
template <class T> void d_Min(T *d_input, tuple2<T, size_t> *d_output, size_t n, int batch = 1);
template <class T> void d_Min(T *d_input, T *d_output, size_t n, int batch = 1);
template <class T> void d_Max(T *d_input, tuple2<T, size_t> *d_output, size_t n, int batch = 1);
template <class T> void d_Max(T *d_input, T *d_output, size_t n, int batch = 1);

//SumMinMax.cu
template <class T> void d_SumMinMax(T* d_input, T* d_sum, T* d_min, T* d_max, size_t n, int batch = 1);

//Dev.cu:
template <class Tmask> void d_Dev(tfloat* d_input, imgstats5* d_output, size_t elements, Tmask* d_mask, int batch = 1);


/////////////////////
//Fourier transform//
/////////////////////

//FFT.cu:
void d_FFTR2C(tfloat* const d_input, tcomplex* const d_output, int const ndimensions, int3 const dimensions, int batch = 1);
void d_FFTR2CFull(tfloat* const d_input, tcomplex* const d_output, int const ndimensions, int3 const dimensions, int batch = 1);
void d_FFTC2C(tcomplex* const d_input, tcomplex* const d_output, int const ndimensions, int3 const dimensions, int batch = 1);
void FFTR2C(tfloat* const h_input, tcomplex* const h_output, int const ndimensions, int3 const dimensions, int batch = 1);
void FFTR2CFull(tfloat* const h_input, tcomplex* const h_output, int const ndimensions, int3 const dimensions, int batch = 1);
void FFTC2C(tcomplex* const h_input, tcomplex* const h_output, int const ndimensions, int3 const dimensions, int batch = 1);

//IFFT.cu:
void d_IFFTC2R(tcomplex* const d_input, tfloat* const d_output, int const ndimensions, int3 const dimensions, int batch = 1);
void d_IFFTC2RFull(tcomplex* const d_input, tfloat* const d_output, int const ndimensions, int3 const dimensions, int batch = 1);
void d_IFFTC2C(tcomplex* const d_input, tcomplex* const d_output, int const ndimensions, int3 const dimensions, int batch = 1);
void IFFTC2R(tcomplex* const h_input, tfloat* const h_output, int const ndimensions, int3 const dimensions, int batch = 1);
void IFFTC2RFull(tcomplex* const h_input, tfloat* const h_output, int const ndimensions, int3 const dimensions, int batch = 1);
void IFFTC2C(tcomplex* const h_input, tcomplex* const h_output, int const ndimensions, int3 const dimensions, int batch = 1);

//HermitianSymmetry.cu:
void d_HermitianSymmetryPad(tcomplex* const d_input, tcomplex* const d_output, int3 const dimensions, int batch = 1);
void d_HermitianSymmetryTrim(tcomplex* const d_input, tcomplex* const d_output, int3 const dimensions, int batch = 1);

//FFTRemap.cu:
void d_RemapFullToHalfFFT(tfloat* d_input, tfloat* d_output, int3 dims);


//////////////////////
//Image Manipulation//
//////////////////////

//Norm.cu:
enum T_NORM_MODE 
{ 
	T_NORM_MEAN01STD = 1, 
	T_NORM_PHASE = 2, 
	T_NORM_STD1 = 3, 
	T_NORM_STD2 = 4, 
	T_NORM_STD3 = 5, 
	T_NORM_OSCAR = 6, 
	T_NORM_CUSTOM = 7 
};
template <class Tmask> void d_Norm(tfloat* d_input, tfloat* d_output, size_t elements, Tmask* d_mask, T_NORM_MODE mode, tfloat scf, int batch = 1);

//Bin.cu:
void d_Bin(tfloat* d_input, tfloat* d_output, int3 dims, int bincount, int batch = 1);

//Bandpass.cu:
void d_Bandpass(tfloat* d_input, tfloat* d_output, int3 dims, tfloat low, tfloat high, tfloat smooth, int batch = 1);

//Coordinates.cu:
void d_Cart2Polar(tfloat* d_input, tfloat* d_output, int2 dims, int batch);
int2 GetCart2PolarSize(int2 dims);


///////////
//Masking//
///////////

//SphereMask.cu:
template <class T> void d_SphereMask(T const* const d_input, T* const d_output, int3 const size, tfloat const* const radius, tfloat const sigma, tfloat3 const* const center, int batch = 1);

//Remap.cu:
template <class T> void d_Remap(T* d_input, intptr_t* d_map, T* d_output, size_t elementsmapped, size_t elementsoriginal, T defvalue, int batch = 1);
template <class T> void Remap(T* h_input, intptr_t* h_map, T* h_output, size_t elementsmapped, size_t elementsoriginal, T defvalue, int batch = 1);
template <class T> void d_MaskSparseToDense(T* d_input, intptr_t** d_mapforward, intptr_t* d_mapbackward, size_t &elementsmapped, size_t elementsoriginal);
template <class T> void MaskSparseToDense(T* h_input, intptr_t** h_mapforward, intptr_t* h_mapbackward, size_t &elementsmapped, size_t elementsoriginal);