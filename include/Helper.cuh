#pragma once
#include "cufft.h"
#include "IO.cuh"
#include "Prerequisites.cuh"


////////////////////
//Helper Functions//
////////////////////

//TypeConversion.cu:

/**
 * \brief Template to convert single-field numerical data to tfloat.
 * \param[in] original	Array in host memory to be converted
 * \param[in] n		Number of elements
 * \returns Array pointer in host memory
 */
template <class T> tfloat* ConvertToTFloat(T const* const original, size_t const n);

/**
 * \brief Template to convert single-field numerical data to tfloat.
 * \param[in] original	Array in host memory to be converted
 * \param[in] copy	Array in host memory that will contain the converted copy
 * \param[in] n		Number of elements
 */
template <class T> void ConvertToTFloat(T const* const original, tfloat* const copy, size_t const n);

/**
 * \brief Template to convert tfloat data to other single-field numerical formats.
 * \param[in] original	Array in host memory to be converted
 * \param[in] n		Number of elements
 * \returns Array pointer in host memory
 */
template <class T> T* ConvertTFloatTo(tfloat const* const original, size_t const n);

/**
 * \brief Template to convert tfloat data to other single-field numerical formats.
 * \param[in] original	Array in host memory to be converted
 * \param[in] copy	Array in host memory that will contain the converted copy
 * \param[in] n		Number of elements
 */
template <class T> void ConvertTFloatTo(tfloat const* const original, T* const copy, size_t const n);


/**
 * \brief Template to convert complex data in split representation (as in Matlab) to interleaved double-field form.
 * \param[in] originalr	Array in host memory with real part of the data to be converted
 * \param[in] originali	Array in host memory with imaginary part of the data to be converted
 * \param[in] n		Number of elements
 * \returns Array pointer in host memory
 */
template <class T> tcomplex* ConvertSplitComplexToTComplex(T const* const originalr, T const* const originali, size_t const n);

/**
 * \brief Template to convert complex data in split representation (as in Matlab) to interleaved double-field form.
 * \param[in] originalr	Array in host memory with real part of the data to be converted
 * \param[in] originali	Array in host memory with imaginary part of the data to be converted
 * \param[in] copy	Array in host memory that will contain the converted copy
 * \param[in] n		Number of elements
 */
template <class T> void ConvertSplitComplexToTComplex(T const* const originalr, T const* const originali, tcomplex* const copy, size_t const n);

/**
 * \brief Template to convert complex data in interleaved double-field form to split representation (as in Matlab).
 * \param[in] original	Array in host memory with tcomplex data to be converted
 * \param[in] n		Number of elements
 * \returns Array pointer in host memory: first half contains real part, second half contains imaginary part
 */
template <class T> T* ConvertTComplexToSplitComplex(tcomplex const* const original, size_t const n);

/**
 * \brief Template to convert complex data in interleaved double-field form to split representation (as in Matlab).
 * \param[in] original	Array in host memory with tcomplex data to be converted
 * \param[in] copyr	Array in host memory that will contain the real part of the converted data
 * \param[in] copyi	Array in host memory that will contain the imaginary part of the converted data
 * \param[in] n		Number of elements
 */
template <class T> void ConvertTComplexToSplitComplex(tcomplex const* const original, T* const copyr, T* const copyi, size_t const n);


/**
 * \brief Template to convert single-field numerical data to tfloat.
 * \param[in] d_original	Array in device memory to be converted
 * \param[in] d_copy	Array in device memory that will contain the converted copy
 * \param[in] n		Number of elements
 */
template <class T> void d_ConvertToTFloat(T const* const d_original, tfloat* const d_copy, size_t const n);

/**
 * \brief Template to convert tfloat data to other single-field numerical formats.
 * \param[in] d_original	Array in device memory to be converted
 * \param[in] d_copy	Array in device memory that will contain the converted copy
 * \param[in] n		Number of elements
 */
template <class T> void d_ConvertTFloatTo(tfloat const* const d_original, T* const d_copy, size_t const n);

/**
 * \brief Template to convert complex data in split representation (as in Matlab) to interleaved double-field form.
 * \param[in] d_originalr	Array in device memory with real part of the data to be converted
 * \param[in] d_originali	Array in device memory with imaginary part of the data to be converted
 * \param[in] d_copy	Array in device memory that will contain the converted copy
 * \param[in] n		Number of elements
 */
template <class T> void d_ConvertSplitComplexToTComplex(T const* const d_originalr, T const* const d_originali, tcomplex* const d_copy, size_t const n);

/**
 * \brief Template to convert complex data in interleaved double-field form to split representation (as in Matlab).
 * \param[in] d_original	Array in device memory with tcomplex data to be converted
 * \param[in] d_copyr	Array in device memory that will contain the real part of the converted data
 * \param[in] d_copyi	Array in device memory that will contain the imaginary part of the converted data
 * \param[in] n		Number of elements
 */
template <class T> void d_ConvertTComplexToSplitComplex(tcomplex const* const d_original, T* const d_copyr, T* const d_copyi, size_t const n);


/**
 * \brief Template to extract and (if needed) convert the real part of tcomplex data.
 * \param[in] d_input	Array in device memory with the tcomplex data
 * \param[in] d_output	Array in device memory that will contain the extracted real part
 * \param[in] n		Number of elements
 */
template <class T> void d_Re(tcomplex const* const d_input, T* const d_output, size_t const n);

/**
 * \brief Template to extract and (if needed) convert the imaginary part of tcomplex data.
 * \param[in] d_input	Array in device memory with the tcomplex data
 * \param[in] d_output	Array in device memory that will contain the extracted imaginary part
 * \param[in] n		Number of elements
 */
template <class T> void d_Im(tcomplex const* const d_input, T* const d_output, size_t const n);

//MDPointer.cu:

/**
 * \brief Class to facilitate handling of identical data across multiple devices
 */
template <class T> class MDPointer
{
	int devicecount;	//!< Number of devices the data is copied across

public:
	T** pointers;	//!< Array of device pointers, nth pointer corresponds to pointer on nth device
	MDPointer();	//!< Constructor
	~MDPointer();	//!< Destructor

	/**
	 * \brief Allocates memory on all devices
	 * \param[in] size	Size of the allocated memory in bytes
	 */
	void Malloc(size_t size);

	void Free();	//<! Frees allocated memory on all devices
	
	/**
	 * \brief Allocates memory on all devices and fills it with data from a host array
	 * \param[in] h_src	Array in host memory from where data should be copied
	 * \param[in] size	Size of the allocated memory and copied data in bytes
	 */
	void MallocFromHostArray(T* h_src, size_t size);
	
	/**
	 * \brief Allocates memory on all devices and fills it with data from a host array
	 * \param[in] h_src	Array in host memory from where data should be copied
	 * \param[in] devicesize	Size of the allocated memory in bytes
	 * \param[in] hostsize	Size of the copied data in bytes
	 */
	void MallocFromHostArray(T* h_src, size_t devicesize, size_t hostsize);
	
	/**
	 * \brief Copies memory from host memory to an offset location in device memory
	 * \param[in] h_src	Array in host memory from where data should be copied
	 * \param[in] deviceoffset	Uniform offset to be applied to all device pointers before copying
	 * \param[in] size	Size of the copied data in bytes
	 */
	void Memcpy(T* h_src, size_t deviceoffset, size_t size);
	
	/**
	 * \brief Allocates memory on all devices and initializes every field with the same value
	 * \param[in] elements	Number of elements
	 * \param[in] value	Value used for initialization
	 */
	void MallocValueFilled(size_t elements, T value);

	bool operator == (const MDPointer<T> &other) const;	//<! Determines if two MDPointers are equal
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

/**
 * \brief Converts a host array in a specified data format to tfloat.
 * \param[in] h_input	Array in host memory to be converted
 * \param[in] datatype		Data type for the host array
 * \param[in] elements		Number of elements
 * \returns Array pointer in host memory
 */
tfloat* MixedToHostTfloat(void* h_input, EM_DATATYPE datatype, size_t elements);

/**
* \brief Converts a host array in a specified data format to tfloat.
* \param[in] h_input	Array in host memory to be converted
* \param[in] datatype		Data type for the host array
* \param[in] elements		Number of elements
* \returns Array pointer in host memory
*/
tfloat* MixedToHostTfloat(void* h_input, MRC_DATATYPE datatype, size_t elements);

void WriteToBinaryFile(string path, void* data, size_t bytes);

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

/**
 * \brief Template to convert data in a host array to a device array of a different data type.
 * \param[in] h_array	Array in host memory to be converted
 * \param[in] elements	Number of elements
 * \returns Array pointer in device memory
 */
template <class T1, class T2> T2* CudaMallocFromHostArrayConverted(T1* h_array, size_t elements);

/**
 * \brief Template to convert data in a host array to a device array of a different data type.
 * \param[in] h_array	Array in host memory to be converted
 * \param[in] d_output	Pointer that will contain the array with the converted data
 * \param[in] elements	Number of elements
 */
template <class T1, class T2> void CudaMallocFromHostArrayConverted(T1* h_array, T2** d_output, size_t elements);

/**
 * \brief Template to convert data in a host array to a device array of a different data type (without allocating new memory).
 * \param[in] h_array	Array in host memory to be converted
 * \param[in] d_output	Array in device memory that will contain the converted data
 * \param[in] elements	Number of elements
 */
template <class T1, class T2> void CudaMemcpyFromHostArrayConverted(T1* h_array, T2* d_output, size_t elements);

/**
 * \brief Creates an array of floats initialized to 0.0f in device memory with the specified element count.
 * \param[in] elements	Element count
 * \returns Array pointer in device memory
 */
float* CudaMallocZeroFilledFloat(size_t elements);

/**
 * \brief Copies elements from source to multiple consecutive destinations.
 * \param[in] dst	Destination start address
 * \param[in] src	Source address
 * \param[in] elements	Number of elements
 * \param[in] copies	Number of copies
 * \returns Array pointer in device memory
 */
template<class T> void CudaMemcpyMulti(T* dst, T* src, uint elements, uint copies, uint batch = 1);

/**
 * \brief Copies elements from source to destination using strided access for both pointers.
 * \param[in] dst	Destination start address
 * \param[in] src	Source address
 * \param[in] elements	Number of elements
 * \param[in] stridedst	Stride in number of elements for destination pointer (1 = no stride)
 * \param[in] stridesrc	Stride in number of elements for source pointer (1 = no stride)
 * \returns Array pointer in device memory
 */
template<class T> void CudaMemcpyStrided(T* dst, T* src, size_t elements, int stridedst, int stridesrc);

/**
 * \brief Creates an array of T initialized to value in device memory with the specified element count.
 * \param[in] elements	Element count
 * \param[in] value		Initial value
 * \returns Array pointer in device memory
 */
template <class T> T* CudaMallocValueFilled(size_t elements, T value);
	
/**
 * \brief Initializes every field of an array with the same value
 * \param[in] d_array	Array in device memory that should be initialized
 * \param[in] elements	Number of elements
 * \param[in] value	Value used for initialization
 */
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

/**
 * \brief Converts a host array in a specified data format to tfloat in device memory.
 * \param[in] h_input	Array in host memory to be converted
 * \param[in] d_output	Array in device memory that will contain the converted data
 * \param[in] datatype		Data type for the host array
 * \param[in] elements		Number of elements
 */
void MixedToDeviceTfloat(void* h_input, tfloat* d_output, EM_DATATYPE datatype, size_t elements);

/**
* \brief Converts a host array in a specified data format to tfloat in device memory.
* \param[in] h_input	Array in host memory to be converted
* \param[in] d_output	Array in device memory that will contain the converted data
* \param[in] datatype		Data type for the host array
* \param[in] elements		Number of elements
*/
void MixedToDeviceTfloat(void* h_input, tfloat* d_output, MRC_DATATYPE datatype, size_t elements);

/**
* \brief Converts a host array in a specified data format to tfloat in device memory.
* \param[in] h_input	Array in host memory to be converted
* \param[in] datatype		Data type for the host array
* \param[in] elements		Number of elements
* \returns Array pointer in device\memory
*/
tfloat* MixedToDeviceTfloat(void* h_input, EM_DATATYPE datatype, size_t elements);

/**
* \brief Converts a host array in a specified data format to tfloat in device memory.
* \param[in] h_input	Array in host memory to be converted
* \param[in] datatype		Data type for the host array
* \param[in] elements		Number of elements
* \returns Array pointer in device\memory
*/
tfloat* MixedToDeviceTfloat(void* h_input, MRC_DATATYPE datatype, size_t elements);

int GetFileSize(string path);
void* MallocFromBinaryFile(string path);
void* CudaMallocFromBinaryFile(string path);
void CudaWriteToBinaryFile(string path, void* d_data, size_t elements);


cudaPitchedPtr CopyVolumeDeviceToDevice(tfloat* d_input, int3 dims);
cudaPitchedPtr CopyVolumeHostToDevice(tfloat* h_input, int3 dims);

//Misc.cu:

/**
 * \brief Integer version of the pow function.
 * \param[in] base	Number to be raised
 * \param[in] exponent	Power to raise the base to
 * \returns base^exponent
 */
 int pow(int base, int exponent);

 //TextureObject.cu:

void d_BindTextureToArray(tfloat* d_input, cudaArray_t &createdarray, cudaTex &createdtexture, int2 dims, cudaTextureFilterMode filtermode, bool normalizedcoords);
void d_BindTextureToArray(tfloat* d_input, cudaArray_t* &h_createdarrays, cudaTex* &h_createdtextures, int2 dims, cudaTextureFilterMode filtermode, bool normalizedcoords, int nimages);
void d_BindTextureTo3DArray(tfloat* d_input, cudaArray_t &createdarray, cudaTex &createdtexture, int3 dims, cudaTextureFilterMode filtermode, bool normalizedcoords);
void d_BindTextureTo3DArray(tfloat* d_input, cudaArray_t* &h_createdarrays, cudaTex* &h_createdtextures, int3 dims, cudaTextureFilterMode filtermode, bool normalizedcoords, int nvolumes);

//KaiserBessel.cpp:
double chebev(double a, double b, double c[], int m, double x);
void beschb(double x, double *gam1, double *gam2, double *gampl, double *gammi);
void bessjy(double x, double xnu, double *rj, double *ry, double *rjp, double *ryp);
double bessj0(double x);
double bessi0(double x);
double bessi1(double x);
double bessi0_5(double x);
double bessi1_5(double x);
double bessi2(double x);
double bessi2_5(double x);
double bessi3(double x);
double bessi3_5(double x);
double bessi4(double x);
double bessj1_5(double x);
double bessj3_5(double x);
double kaiser_Fourier_value(double w, double a, double alpha, int m);

//GridSpacing.cu:
int3* GetEqualGridSpacing(int2 dimsimage, int2 dimsregion, float overlapfraction, int2 &dimsgrid);