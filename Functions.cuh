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

template <class T> void d_Re(tcomplex const* const d_input, T* const d_output, size_t const n);
template <class T> void d_Im(tcomplex const* const d_input, T* const d_output, size_t const n);

//MDPointer.cu:

template <class T> class MDPointer
{
	int devicecount;

public:
	T** pointers;
	MDPointer();
	~MDPointer();
	void Malloc(size_t size);
	void Free();
	void MallocFromHostArray(T* h_src, size_t size);
	void MallocFromHostArray(T* h_src, size_t devicesize, size_t hostsize);
	void Memcpy(T* h_src, size_t deviceoffset, size_t size);
	void MallocValueFilled(size_t elements, T value);
	bool operator == (const MDPointer<T> &other) const;
};
template class MDPointer<float>;
template class MDPointer<double>;
template class MDPointer<tcomplex>;
template class MDPointer<char>;
template class MDPointer<int>;
template class MDPointer<short>;
template class MDPointer<bool>;
template class MDPointer<tfloat2>;
template class MDPointer<tfloat3>;
template class MDPointer<tfloat4>;

//Memory.cu:

/**
 * \brief Allocates an array in host memory that is a copy of the array in device memory.
 * \param[in] d_array	Array in device memory to be copied
 * \param[in] size		Array size in bytes
 * \returns Array pointer in host memory
 */
void* MallocFromDeviceArray(void* d_array, size_t size);

/**
 * \brief Allocates an array in pinned host memory that is a copy of the array in device memory.
 * \param[in] d_array	Array in device memory to be copied
 * \param[in] size		Array size in bytes
 * \returns Array pointer in host memory
 */
void* MallocPinnedFromDeviceArray(void* d_array, size_t size);

tfloat* MixedToHostTfloat(void* h_input, EM_DATATYPE datatype, size_t elements);

/**
 * \brief Allocates an array in device memory with an alignment constraint (useful for odd-sized 2D textures).
 * \param[in] widthbytes	Array width in bytes
 * \param[in] height		Array height
 * \param[in] pitch			Pitched width value will be written here
 * \param[in] alignment		Alignment constraint, usually 32 bytes
 * \returns Array pointer in device memory
 */
void* CudaMallocAligned2D(size_t widthbytes, size_t height, int* pitch, int alignment = 32);

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

template <class T1, class T2> T2* CudaMallocFromHostArrayConverted(T1* h_array, size_t elements);
template <class T1, class T2> void CudaMemcpyFromHostArrayConverted(T1* h_array, T2* d_output, size_t elements);
template <class T1, class T2> void CudaMallocFromHostArrayConverted(T1* h_array, T2** d_output, size_t elements);

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

template <class T> void d_ValueFill(T* d_array, size_t elements, T value);

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


void MixedToDeviceTfloat(void* h_input, tfloat* d_output, EM_DATATYPE datatype, size_t elements);
tfloat* MixedToDeviceTfloat(void* h_input, EM_DATATYPE datatype, size_t elements);

//Misc.cu:
 int pow(int base, int exponent);


 //////
 //IO//
 //////

 //mrc.cu:
void ReadMRC(string path, void** data, EM_DATATYPE &datatype, int nframe = 0, bool flipx = false);
void ReadMRCDims(string path, int3 &dims);

//em.cu:
void ReadEM(string path, void** data, EM_DATATYPE &datatype, int nframe = 0);
void ReadEMDims(string path, int3 &dims);

//raw.cu:
void ReadRAW(string path, void** data, EM_DATATYPE datatype, int3 dims, int nframe = 0, size_t headerbytes = 0);

////////////
//Generics//
////////////

//Arithmetics.cu:
template <class T> void d_MultiplyByScalar(T* d_input, T* d_output, size_t elements, T multiplicator);
template <class T> void d_MultiplyByScalar(T* d_input, T* d_multiplicators, T* d_output, size_t elements, int batch = 1);
template <class T> void d_MultiplyByVector(T* d_input, T* d_multiplicators, T* d_output, size_t elements, int batch = 1);

void d_ComplexMultiplyByVector(tcomplex* d_input, tfloat* d_multiplicators, tcomplex* d_output, size_t elements, int batch = 1);
void d_ComplexMultiplyByScalar(tcomplex* d_input, tcomplex* d_output, size_t elements, tfloat multiplicator);
void d_ComplexMultiplyByScalar(tcomplex* d_input, tfloat* d_multiplicators, tcomplex* d_output, size_t elements, int batch = 1);

void d_ComplexMultiplyByVector(tcomplex* d_input, tcomplex* d_multiplicators, tcomplex* d_output, size_t elements, int batch = 1);
void d_ComplexMultiplyByScalar(tcomplex* d_input, tcomplex* d_output, size_t elements, tcomplex multiplicator);
void d_ComplexMultiplyByScalar(tcomplex* d_input, tcomplex* d_multiplicators, tcomplex* d_output, size_t elements, int batch = 1);

void d_ComplexMultiplyByConjVector(tcomplex* d_input, tcomplex* d_multiplicators, tcomplex* d_output, size_t elements, int batch = 1);
void d_ComplexMultiplyByConjScalar(tcomplex* d_input, tcomplex* d_output, size_t elements, tcomplex multiplicator);
void d_ComplexMultiplyByConjScalar(tcomplex* d_input, tcomplex* d_multiplicators, tcomplex* d_output, size_t elements, int batch = 1);

template <class T> void d_AddScalar(T* d_input, T* d_output, size_t elements, T summand);
template <class T> void d_AddScalar(T* d_input, T* d_summands, T* d_output, size_t elements, int batch = 1);
template <class T> void d_AddVector(T* d_input, T* d_summands, T* d_output, size_t elements, int batch = 1);

template <class T> void d_SubtractScalar(T* d_input, T* d_output, size_t elements, T subtrahend);
template <class T> void d_SubtractScalar(T* d_input, T* d_subtrahends, T* d_output, size_t elements, int batch = 1);
template <class T> void d_SubtractVector(T* d_input, T* d_subtrahends, T* d_output, size_t elements, int batch = 1);

template <class T> void d_Sqrt(T* d_input, T* d_output, size_t elements);
template <class T> void d_Square(T* d_input, T* d_output, size_t elements, int batch = 1);
template <class T> void d_Pow(T* d_input, T* d_output, size_t elements, T exponent);
template <class T> void d_Abs(T* d_input, T* d_output, size_t elements);

template <class T> void d_MaxOp(T* d_input1, T* d_input2, T* d_output, size_t elements);
template <class T> void d_MinOp(T* d_input1, T* d_input2, T* d_output, size_t elements);

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

//MinMaxMonolithic.cu:
template <class T> void d_MinMonolithic(T* d_input, tuple2<T, size_t>* d_output, int n, int batch);
template <class T> void d_MinMonolithic(T* d_input, T* d_output, int n, int batch);
template <class T> void d_MaxMonolithic(T* d_input, tuple2<T, size_t>* d_output, int n, int batch);
template <class T> void d_MaxMonolithic(T* d_input, T* d_output, int n, int batch);

//SumMinMax.cu:
template <class T> void d_SumMinMax(T* d_input, T* d_sum, T* d_min, T* d_max, size_t n, int batch = 1);

//Dev.cu:
template <class Tmask> void d_Dev(tfloat* d_input, imgstats5* d_output, size_t elements, Tmask* d_mask, int batch = 1);

//Extraction.cu:
glm::mat4 GetTransform2D(tfloat2 scale, tfloat rotation, tfloat2 translation);
template <class T> void d_Extract(T* d_input, T* d_output, int3 sourcedims, int3 regiondims, int3 regioncenter, int batch = 1);
void d_Extract2DTransformed(tfloat* d_input, tfloat* d_output, int3 sourcedims, int3 regiondims, tfloat2* h_scale, tfloat* h_rotation, tfloat2* h_translation, T_INTERP_MODE mode, int batch = 1);

//Padding.cu:
enum T_PAD_MODE 
{ 
	T_PAD_VALUE = 1,
	T_PAD_MIRROR = 2,
	T_PAD_TILE = 3
};
template <class T> void d_Pad(T* d_input, T* d_output, int3 inputdims, int3 outputdims, T_PAD_MODE mode, T value, int batch = 1);


/////////////
//Alignment//
/////////////

//Align2D.cu:
enum T_ALIGN_MODE
{
	T_ALIGN_ROT = 1 << 0,
	T_ALIGN_TRANS = 1 << 1,
	T_ALIGN_BOTH = 3
};
void d_Align2D(tfloat* d_input, tfloat* d_targets, int3 dims, int numtargets, tfloat3* d_params, int* d_membership, tfloat* d_scores, int maxtranslation, tfloat maxrotation, int iterations, T_ALIGN_MODE mode, int batch);


///////////////////////
//Binary manipulation//
///////////////////////

template <class T> void d_Dilate(T* d_input, T* d_output, int3 dims, int batch = 1);
template <class T> void d_Binarize(tfloat* d_input, T* d_output, size_t elements, tfloat threshold, int batch = 1);


///////////////
//Correlation//
///////////////

//CCF.cu:
template<class T> void d_CCF(tfloat* d_input1, tfloat* d_input2, tfloat* d_output, int3 dims, bool normalized, T* d_mask, int batch = 1);
template<class T> void d_CCFUnshifted(tfloat* d_input1, tfloat* d_input2, tfloat* d_output, int3 dims, bool normalized, T* d_mask, int batch = 1);

//Peak.cu:
enum T_PEAK_MODE
{
	T_PEAK_INTEGER = 1,
	T_PEAK_SUBCOARSE = 2,
	T_PEAK_SUBFINE = 3
};
void d_Peak(tfloat* d_input, tfloat3* d_positions, tfloat* d_values, int3 dims, T_PEAK_MODE mode, int batch = 1);
void d_LocalPeaks(tfloat* d_input, int3** h_peaks, int* h_peaksnum, int3 dims, int localextent, tfloat threshold, int batch = 1);


///////////////////////
//Cubic interpolation//
///////////////////////

template<class T> void d_CubicBSplinePrefilter2D(T* image, int pitch, int2 dims);
template<class T> void d_CubicBSplinePrefilter3D(T* d_volume, int pitch, int width, int height, int depth);


/////////////////////
//Fourier transform//
/////////////////////

//FFT.cu:
void d_FFTR2C(tfloat* const d_input, tcomplex* const d_output, int const ndimensions, int3 const dimensions, int batch = 1);
void d_FFTR2C(tfloat* const d_input, tcomplex* const d_output, cufftHandle* plan);
void d_IFFTZ2D(cufftDoubleComplex* const d_input, double* const d_output, int const ndimensions, int3 const dimensions, int batch = 1);
void d_FFTR2CFull(tfloat* const d_input, tcomplex* const d_output, int const ndimensions, int3 const dimensions, int batch = 1);
void d_FFTC2C(tcomplex* const d_input, tcomplex* const d_output, int const ndimensions, int3 const dimensions, int batch = 1);
void FFTR2C(tfloat* const h_input, tcomplex* const h_output, int const ndimensions, int3 const dimensions, int batch = 1);
void FFTR2CFull(tfloat* const h_input, tcomplex* const h_output, int const ndimensions, int3 const dimensions, int batch = 1);
void FFTC2C(tcomplex* const h_input, tcomplex* const h_output, int const ndimensions, int3 const dimensions, int batch = 1);

cufftHandle d_FFTR2CGetPlan(int const ndimensions, int3 const dimensions, int batch = 1);

//IFFT.cu:
void d_IFFTC2R(tcomplex* const d_input, tfloat* const d_output, int const ndimensions, int3 const dimensions, int batch = 1);
void d_IFFTC2R(tcomplex* const d_input, tfloat* const d_output, cufftHandle* plan, int3 const dimensions, int batch = 1);
void d_IFFTC2RFull(tcomplex* const d_input, tfloat* const d_output, int const ndimensions, int3 const dimensions, int batch = 1);
void d_IFFTC2C(tcomplex* const d_input, tcomplex* const d_output, int const ndimensions, int3 const dimensions, int batch = 1);
void d_IFFTC2C(tcomplex* const d_input, tcomplex* const d_output, cufftHandle* plan, int3 const dimensions);
void IFFTC2R(tcomplex* const h_input, tfloat* const h_output, int const ndimensions, int3 const dimensions, int batch = 1);
void IFFTC2RFull(tcomplex* const h_input, tfloat* const h_output, int const ndimensions, int3 const dimensions, int batch = 1);
void IFFTC2C(tcomplex* const h_input, tcomplex* const h_output, int const ndimensions, int3 const dimensions, int batch = 1);

cufftHandle d_IFFTC2RGetPlan(int const ndimensions, int3 const dimensions, int batch = 1);
cufftHandle d_IFFTC2CGetPlan(int const ndimensions, int3 const dimensions, int batch = 1);

//HermitianSymmetry.cu:
void d_HermitianSymmetryPad(tcomplex* const d_input, tcomplex* const d_output, int3 const dimensions, int batch = 1);
void d_HermitianSymmetryTrim(tcomplex* const d_input, tcomplex* const d_output, int3 const dimensions, int batch = 1);

//FFTRemap.cu:
template <class T> void d_RemapFull2HalfFFT(T* d_input, T* d_output, int3 dims, int batch = 1);
template <class T> void d_RemapFullFFT2Full(T* d_input, T* d_output, int3 dims, int batch = 1);
template <class T> void d_RemapFull2FullFFT(T* d_input, T* d_output, int3 dims, int batch = 1);

//FFTResize.cu:
void d_FFTCrop(tcomplex* d_input, tcomplex* d_output, int3 olddims, int3 newdims, int batch = 1);
void d_FFTFullCrop(tcomplex* d_input, tcomplex* d_output, int3 olddims, int3 newdims, int batch = 1);
void d_FFTPad(tcomplex* d_input, tcomplex* d_output, int3 olddims, int3 newdims, int batch = 1);
void d_FFTFullPad(tcomplex* d_input, tcomplex* d_output, int3 olddims, int3 newdims, int batch = 1);


//////////////////////
//Image Manipulation//
//////////////////////

//Norm.cu:
enum T_NORM_MODE 
{ 
	T_NORM_NONE = 0,
	T_NORM_MEAN01STD = 1, 
	T_NORM_PHASE = 2, 
	T_NORM_STD1 = 3, 
	T_NORM_STD2 = 4, 
	T_NORM_STD3 = 5, 
	T_NORM_OSCAR = 6, 
	T_NORM_CUSTOM = 7 
};
template <class Tmask> void d_Norm(tfloat* d_input, tfloat* d_output, size_t elements, Tmask* d_mask, T_NORM_MODE mode, tfloat scf, int batch = 1);
void d_NormMonolithic(tfloat* d_input, tfloat* d_output, size_t elements, T_NORM_MODE mode, int batch);

//Bandpass.cu:
void d_Bandpass(tfloat* d_input, tfloat* d_output, int3 dims, tfloat low, tfloat high, tfloat smooth, int batch = 1);
void d_BandpassNeat(tfloat* d_input, tfloat* d_output, int3 dims, tfloat low, tfloat high, tfloat smooth, int batch = 1);

//Xray.cu:
void d_Xray(tfloat* d_input, tfloat* d_output, int3 dims, tfloat ndev = (tfloat)4.6, int region = 2, int batch = 1);


///////////
//Masking//
///////////

//SphereMask.cu:
template <class T> void d_SphereMask(T const* const d_input, T* const d_output, int3 const size, tfloat const* const radius, tfloat const sigma, tfloat3 const* const center, int batch = 1);

//RectangleMask.cu:
template <class T> void d_RectangleMask(T const* const d_input, T* const d_output, int3 const size, int3 const rectsize, tfloat const sigma, int3 const* const center, int batch);

//Remap.cu:
template <class T> void d_Remap(T* d_input, intptr_t* d_map, T* d_output, size_t elementsmapped, size_t elementsoriginal, T defvalue, int batch = 1);
template <class T> void d_RemapReverse(T* d_input, intptr_t* d_map, T* d_output, size_t elementsmapped, size_t elementsdestination, T defvalue, int batch = 1);
template <class T> void Remap(T* h_input, intptr_t* h_map, T* h_output, size_t elementsmapped, size_t elementsoriginal, T defvalue, int batch = 1);
template <class T> void d_MaskSparseToDense(T* d_input, intptr_t** d_mapforward, intptr_t* d_mapbackward, size_t &elementsmapped, size_t elementsoriginal);
template <class T> void MaskSparseToDense(T* h_input, intptr_t** h_mapforward, intptr_t* h_mapbackward, size_t &elementsmapped, size_t elementsoriginal);


//////////////
//Projection//
//////////////

//Backward.cu:
void d_ProjBackward(tfloat* d_volume, int3 dimsvolume, tfloat* d_image, int3 dimsimage, tfloat2* angles, tfloat* weight, int batch = 1);

//Forward.cu:
void d_ProjForward(tfloat* d_volume, int3 dimsvolume, tfloat* d_image, int3 dimsimage, tfloat2* angles, int batch = 1);


//////////////////
//Transformation//
//////////////////

//Bin.cu:
void d_Bin(tfloat* d_input, tfloat* d_output, int3 dims, int bincount, int batch = 1);

//Coordinates.cu:
void d_Cart2Polar(tfloat* d_input, tfloat* d_output, int2 dims, T_INTERP_MODE interpolation, int batch = 1);
void d_CartAtlas2Polar(tfloat* d_input, tfloat* d_output, tfloat2* d_offsets, int2 atlasdims, int2 dims, T_INTERP_MODE interpolation, int batch);
int2 GetCart2PolarSize(int2 dims);

//Shift.cu:
void d_Shift(tfloat* d_input, tfloat* d_output, int3 dims, tfloat3* delta, cufftHandle* planforw = NULL, cufftHandle* planback = NULL, tcomplex* d_sharedintermediate = NULL, int batch = 1);

//Scale.cu:
void d_Scale(tfloat* d_input, tfloat* d_output, int3 olddims, int3 newdims, T_INTERP_MODE mode, cufftHandle* planforw = NULL, cufftHandle* planback = NULL, int batch = 1);
