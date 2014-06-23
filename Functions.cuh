#pragma once
#include "cufft.h"
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
template<class T> void CudaMemcpyMulti(T* dst, T* src, size_t elements, int copies);

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
 * \param[in] datatype		Data type for the host array
 * \param[in] elements		Number of elements
 * \returns Array pointer in device\memory
 */
tfloat* MixedToDeviceTfloat(void* h_input, EM_DATATYPE datatype, size_t elements);

//Misc.cu:

/**
 * \brief Integer version of the pow function.
 * \param[in] base	Number to be raised
 * \param[in] exponent	Power to raise the base to
 * \returns base^exponent
 */
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

/**
 * \brief Multiplies every input element by the same scalar.
 * \param[in] d_input	Array with numbers to be multiplied
 * \param[in] d_output	Array that will contain the result; d_output == d_input is valid
 * \param[in] elements	Number of elements
 * \param[in] multiplicator	Multiplicator used for every operation
 */
template <class T> void d_MultiplyByScalar(T* d_input, T* d_output, size_t elements, T multiplicator);

/**
 * \brief Multiplies every element of the nth vector by the nth scalar
 * \param[in] d_input	Array with numbers to be multiplied
 * \param[in] d_multiplicators	Array with scalar multiplicators for the corresponding vectors in d_input
 * \param[in] d_output	Array that will contain the result; d_output == d_input is valid
 * \param[in] elements	Number of elements per vector
 * \param[in] batch	Number of vectors to be multiplied
 */
template <class T> void d_MultiplyByScalar(T* d_input, T* d_multiplicators, T* d_output, size_t elements, int batch = 1);

/**
 * \brief Performs element-wise multiplication of two vectors
 * \param[in] d_input	Array with numbers to be multiplied
 * \param[in] d_multiplicators	Array with multiplicators for the corresponding elements in d_input
 * \param[in] d_output	Array that will contain the result; d_output == d_input is valid
 * \param[in] elements	Number of elements per vector
 * \param[in] batch	Number of vectors to be multiplied
 */
template <class T> void d_MultiplyByVector(T* d_input, T* d_multiplicators, T* d_output, size_t elements, int batch = 1);


/**
 * \brief Multiplies every input element by the same non-complex scalar.
 * \param[in] d_input	Array with numbers to be multiplied
 * \param[in] d_output	Array that will contain the result; d_output == d_input is valid
 * \param[in] elements	Number of elements
 * \param[in] multiplicator	Multiplicator used for every operation
 */
void d_ComplexMultiplyByScalar(tcomplex* d_input, tcomplex* d_output, size_t elements, tfloat multiplicator);

/**
 * \brief Multiplies every element of the nth vector by the nth non-complex scalar
 * \param[in] d_input	Array with numbers to be multiplied
 * \param[in] d_multiplicators	Array with scalar multiplicators for the corresponding vectors in d_input
 * \param[in] d_output	Array that will contain the result; d_output == d_input is valid
 * \param[in] elements	Number of elements per vector
 * \param[in] batch	Number of vectors to be multiplied
 */
void d_ComplexMultiplyByScalar(tcomplex* d_input, tfloat* d_multiplicators, tcomplex* d_output, size_t elements, int batch = 1);

/**
 * \brief Performs element-wise multiplication of a complex vector by a non-complex vector
 * \param[in] d_input	Array with numbers to be multiplied
 * \param[in] d_multiplicators	Array with multiplicators for the corresponding elements in d_input
 * \param[in] d_output	Array that will contain the result; d_output == d_input is valid
 * \param[in] elements	Number of elements per vector
 * \param[in] batch	Number of vectors to be multiplied
 */
void d_ComplexMultiplyByVector(tcomplex* d_input, tfloat* d_multiplicators, tcomplex* d_output, size_t elements, int batch = 1);


/**
 * \brief Multiplies every input element by the same complex scalar.
 * \param[in] d_input	Array with numbers to be multiplied
 * \param[in] d_output	Array that will contain the result; d_output == d_input is valid
 * \param[in] elements	Number of elements
 * \param[in] multiplicator	Multiplicator used for every operation
 */
void d_ComplexMultiplyByScalar(tcomplex* d_input, tcomplex* d_output, size_t elements, tcomplex multiplicator);

/**
 * \brief Multiplies every element of the nth vector by the nth complex scalar
 * \param[in] d_input	Array with numbers to be multiplied
 * \param[in] d_multiplicators	Array with scalar multiplicators for the corresponding vectors in d_input
 * \param[in] d_output	Array that will contain the result; d_output == d_input is valid
 * \param[in] elements	Number of elements per vector
 * \param[in] batch	Number of vectors to be multiplied
 */
void d_ComplexMultiplyByScalar(tcomplex* d_input, tcomplex* d_multiplicators, tcomplex* d_output, size_t elements, int batch = 1);

/**
 * \brief Performs element-wise multiplication of two complex vectors
 * \param[in] d_input	Array with numbers to be multiplied
 * \param[in] d_multiplicators	Array with multiplicators for the corresponding elements in d_input
 * \param[in] d_output	Array that will contain the result; d_output == d_input is valid
 * \param[in] elements	Number of elements per vector
 * \param[in] batch	Number of vectors to be multiplied
 */
void d_ComplexMultiplyByVector(tcomplex* d_input, tcomplex* d_multiplicators, tcomplex* d_output, size_t elements, int batch = 1);


/**
 * \brief Multiplies every input element by the conjugate of the same complex scalar.
 * \param[in] d_input	Array with numbers to be multiplied
 * \param[in] d_output	Array that will contain the result; d_output == d_input is valid
 * \param[in] elements	Number of elements
 * \param[in] multiplicator	Multiplicator used for every operation
 */
void d_ComplexMultiplyByConjScalar(tcomplex* d_input, tcomplex* d_output, size_t elements, tcomplex multiplicator);

/**
 * \brief Multiplies every element of the nth vector by the conjugate of the nth complex scalar
 * \param[in] d_input	Array with numbers to be multiplied
 * \param[in] d_multiplicators	Array with scalar multiplicators for the corresponding vectors in d_input
 * \param[in] d_output	Array that will contain the result; d_output == d_input is valid
 * \param[in] elements	Number of elements per vector
 * \param[in] batch	Number of vectors to be multiplied
 */
void d_ComplexMultiplyByConjScalar(tcomplex* d_input, tcomplex* d_multiplicators, tcomplex* d_output, size_t elements, int batch = 1);

/**
 * \brief Performs element-wise multiplication of a complex vector by the conjugate of another complex vector
 * \param[in] d_input	Array with numbers to be multiplied
 * \param[in] d_multiplicators	Array with multiplicators for the corresponding elements in d_input
 * \param[in] d_output	Array that will contain the result; d_output == d_input is valid
 * \param[in] elements	Number of elements per vector
 * \param[in] batch	Number of vectors to be multiplied
 */
void d_ComplexMultiplyByConjVector(tcomplex* d_input, tcomplex* d_multiplicators, tcomplex* d_output, size_t elements, int batch = 1);


/**
 * \brief Divides every input element by the same scalar.
 * \param[in] d_input	Array with numbers to be multiplied
 * \param[in] d_output	Array that will contain the result; d_output == d_input is valid
 * \param[in] elements	Number of elements
 * \param[in] divisor	Divisor used for every operation
 */
template <class T> void d_DivideByScalar(T* d_input, T* d_output, size_t elements, T divisor);

/**
 * \brief Divides every element of the nth vector by the nth scalar
 * \param[in] d_input	Array with numbers to be multiplied
 * \param[in] d_divisors	Array with scalar divisors for the corresponding vectors in d_input
 * \param[in] d_output	Array that will contain the result; d_output == d_input is valid
 * \param[in] elements	Number of elements per vector
 * \param[in] batch	Number of vectors to be divided
 */
template <class T> void d_DivideByScalar(T* d_input, T* d_divisors, T* d_output, size_t elements, int batch = 1);

/**
 * \brief Performs element-wise multiplication of two vectors
 * \param[in] d_input	Array with numbers to be multiplied
 * \param[in] d_divisors	Array with divisors for the corresponding elements in d_input
 * \param[in] d_output	Array that will contain the result; d_output == d_input is valid
 * \param[in] elements	Number of elements per vector
 * \param[in] batch	Number of vectors to be divided
 */
template <class T> void d_DivideByVector(T* d_input, T* d_divisors, T* d_output, size_t elements, int batch = 1);

/**
 * \brief Performs element-wise multiplication of two vectors; if division by 0 occurs, 0 is written to the result instead of NaN
 * \param[in] d_input	Array with numbers to be multiplied
 * \param[in] d_divisors	Array with divisors for the corresponding elements in d_input
 * \param[in] d_output	Array that will contain the result; d_output == d_input is valid
 * \param[in] elements	Number of elements per vector
 * \param[in] batch	Number of vectors to be divided
 */
template <class T> void d_DivideSafeByVector(T* d_input, T* d_divisors, T* d_output, size_t elements, int batch = 1);


/**
 * \brief Adds the same scalar to every input element.
 * \param[in] d_input	Array with numbers to be incremented
 * \param[in] d_output	Array that will contain the result; d_output == d_input is valid
 * \param[in] elements	Number of elements
 * \param[in] summand	Summand used for every operation
 */
template <class T> void d_AddScalar(T* d_input, T* d_output, size_t elements, T summand);

/**
 * \brief Adds the nth summand to all elements of the nth input vector
 * \param[in] d_input	Array with numbers to be incremented
 * \param[in] d_summands	Array with scalar summands for the corresponding vectors in d_input
 * \param[in] d_output	Array that will contain the result; d_output == d_input is valid
 * \param[in] elements	Number of elements per vector
 * \param[in] batch	Number of vectors to be incremented
 */
template <class T> void d_AddScalar(T* d_input, T* d_summands, T* d_output, size_t elements, int batch = 1);

/**
 * \brief Adds a vector to all input vectors
 * \param[in] d_input	Array with numbers to be incremented
 * \param[in] d_summands	Array with vector summand
 * \param[in] d_output	Array that will contain the result; d_output == d_input is valid
 * \param[in] elements	Number of elements per vector
 * \param[in] batch	Number of vectors to be incremented
 */
template <class T> void d_AddVector(T* d_input, T* d_summands, T* d_output, size_t elements, int batch = 1);


/**
 * \brief Subtracts the same scalar from every input element.
 * \param[in] d_input	Array with numbers to be decremented
 * \param[in] d_output	Array that will contain the result; d_output == d_input is valid
 * \param[in] elements	Number of elements
 * \param[in] subtrahend	Subtrahend used for every operation
 */
template <class T> void d_SubtractScalar(T* d_input, T* d_output, size_t elements, T subtrahend);

/**
 * \brief Subtracts the nth subtrahend from all elements of the nth input vector
 * \param[in] d_input	Array with numbers to be decremented
 * \param[in] d_subtrahends	Array with scalar subtrahends for the corresponding vectors in d_input
 * \param[in] d_output	Array that will contain the result; d_output == d_input is valid
 * \param[in] elements	Number of elements per vector
 * \param[in] batch	Number of vectors to be decremented
 */
template <class T> void d_SubtractScalar(T* d_input, T* d_subtrahends, T* d_output, size_t elements, int batch = 1);

/**
 * \brief Subtracts a vector from all input vectors
 * \param[in] d_input	Array with numbers to be decremented
 * \param[in] d_subtrahends	Array with vector subtrahend
 * \param[in] d_output	Array that will contain the result; d_output == d_input is valid
 * \param[in] elements	Number of elements per vector
 * \param[in] batch	Number of vectors to be decremented
 */
template <class T> void d_SubtractVector(T* d_input, T* d_subtrahends, T* d_output, size_t elements, int batch = 1);


/**
 * \brief Computes the square root of every input element
 * \param[in] d_input	Array with numbers to be raised to ^1/2
 * \param[in] d_output	Array that will contain the result; d_output == d_input is valid
 * \param[in] elements	Number of elements
 */
template <class T> void d_Sqrt(T* d_input, T* d_output, size_t elements);

/**
 * \brief Computes the square of every input element
 * \param[in] d_input	Array with numbers to be raised to ^2
 * \param[in] d_output	Array that will contain the result; d_output == d_input is valid
 * \param[in] elements	Number of elements
 */
template <class T> void d_Square(T* d_input, T* d_output, size_t elements);

/**
 * \brief Raises every input element to the same power
 * \param[in] d_input	Array with numbers to be raised
 * \param[in] d_output	Array that will contain the result; d_output == d_input is valid
 * \param[in] elements	Number of elements
 * \param[in] exponent	Power to raise every element to
 */
template <class T> void d_Pow(T* d_input, T* d_output, size_t elements, T exponent);

/**
 * \brief Computes the absolute value (magnitude) of every input element
 * \param[in] d_input	Array with numbers to be made absolute
 * \param[in] d_output	Array that will contain the result; d_output == d_input is valid
 * \param[in] elements	Number of elements
 */
template <class T> void d_Abs(T* d_input, T* d_output, size_t elements);

/**
 * \brief Computes the inverse (1/value) of every input element
 * \param[in] d_input	Array with numbers to be inversed
 * \param[in] d_output	Array that will contain the result; d_output == d_input is valid
 * \param[in] elements	Number of elements
 */
template <class T> void d_Inv(T* d_input, T* d_output, size_t elements);


/**
 * \brief Transforms every complex input element from polar to cartesian form
 * \param[in] d_input	Array with numbers in polar form
 * \param[in] d_cart	Array that will contain the cartesian form; d_output == d_input is valid
 * \param[in] elements	Number of elements
 */
void d_ComplexPolarToCart(tcomplex* d_polar, tcomplex* d_cart, size_t elements);

/**
 * \brief Transforms every complex input element from cartesian to polar form
 * \param[in] d_input	Array with numbers in cartesian form
 * \param[in] d_cart	Array that will contain the polar form; d_output == d_input is valid
 * \param[in] elements	Number of elements
 */
void d_ComplexCartToPolar(tcomplex* d_cart, tcomplex* d_polar, size_t elements);


/**
 * \brief For each pair n of input elements, max(input1[n], input2[n]) is written to output
 * \param[in] d_input1	Array with the first numbers in each pair
 * \param[in] d_input2	Array with the second numbers in each pair
 * \param[in] d_output	Array that will contain the result; d_output == d_input is valid
 * \param[in] elements	Number of elements
 */
template <class T> void d_MaxOp(T* d_input1, T* d_input2, T* d_output, size_t elements);

/**
 * \brief For each pair n of input elements, min(input1[n], input2[n]) is written to output
 * \param[in] d_input1	Array with the first numbers in each pair
 * \param[in] d_input2	Array with the second numbers in each pair
 * \param[in] d_output	Array that will contain the result; d_output == d_input is valid
 * \param[in] elements	Number of elements
 */
template <class T> void d_MinOp(T* d_input1, T* d_input2, T* d_output, size_t elements);


/**
 * \brief Computes the smallest power of 2 that is >= x
 * \param[in] x	Lower limit for the power of 2
 * \returns Next power of 2
 */
size_t NextPow2(size_t x);

/**
 * \brief Determines if a number is a power of 2
 * \param[in] x	Number in question
 * \returns True if x is power of 2
 */
bool IsPow2(size_t x);

//CompositeArithmetics.cu:

/**
 * \brief Computes (n - a)^2 for every input element n and the fixed scalar a
 * \param[in] d_input	Array with input numbers
 * \param[in] d_output	Array that will contain the result; d_output == d_input is valid
 * \param[in] elements	Number of elements
 * \param[in] scalar	The scalar substracted from every element before squaring
 */
template <class T> void d_SquaredDistanceFromScalar(T* d_input, T* d_output, size_t elements, T scalar);

/**
 * \brief Computes (n - a)^2 for every element n in input vector and the per-vector fixed scalar a
 * \param[in] d_input	Array with input numbers
 * \param[in] d_scalars	The scalar substracted from every vector element before squaring
 * \param[in] d_output	Array that will contain the result; d_output == d_input is valid
 * \param[in] elements	Number of elements
 * \param[in] batch	Number of vectors
 */
template <class T> void d_SquaredDistanceFromScalar(T* d_input, T* d_scalars, T* d_output, size_t elements, int batch = 1);

/**
 * \brief Computes (n - a).^2 for every input vector n and fixed vector a (.^2 = element-wise square)
 * \param[in] d_input	Array with input numbers
 * \param[in] d_vector	The vector substracted from every input vector before squaring
 * \param[in] d_output	Array that will contain the result; d_output == d_input is valid
 * \param[in] elements	Number of elements
 * \param[in] batch	Number of vectors
 */
template <class T> void d_SquaredDistanceFromVector(T* d_input, T* d_vector, T* d_output, size_t elements, int batch = 1);

//Histogram.cu:

/**
 * \brief Bins input elements fitting [minval; maxval] into n bins and outputs bin sizes; bin centers are (n + 0.5) * (maxval - minval) / nbins + minval
 * \param[in] d_input	Array with input numbers
 * \param[in] d_histogram	Array that will contain the histogram; d_histogram == d_input is not valid
 * \param[in] elements	Number of elements
 * \param[in] nbins	Number of bins
 * \param[in] minval	Elements >= minval are considered
 * \param[in] maxval	Elements <= maxval are considered
 * \param[in] batch	Number of input populations/histograms
 */
template<class T> void d_Histogram(T* d_input, uint* d_histogram, size_t elements, int nbins, T minval, T maxval, int batch = 1);

//IndexOf.cu:

/**
 * \brief Finds the position of the first occurrence of a value; if no integer positions match, linear interpolation is performed
 * \param[in] d_input	Array with input numbers
 * \param[in] d_output	Array that will contain the position; 0 if all values are smaller, elements if all values are larger, -1 if NaN is encountered
 * \param[in] elements	Number of elements
 * \param[in] value	Value to be found
 * \param[in] mode	Interpolation mode; only T_INTERP_LINEAR is supported
 * \param[in] batch	Number of input vectors/output positions
 */
void d_FirstIndexOf(tfloat* d_input, tfloat* d_output, size_t elements, tfloat value, T_INTERP_MODE mode, int batch = 1);

/**
 * \brief Finds the position of the first (local) minimum
 * \param[in] d_input	Array with input numbers
 * \param[in] d_output	Array that will contain the position; -1 if no minimum is found
 * \param[in] elements	Number of elements
 * \param[in] mode	Interpolation mode; only T_INTERP_LINEAR is supported (thus useless)
 * \param[in] batch	Number of input vectors/output positions
 */
void d_FirstMinimum(tfloat* d_input, tfloat* d_output, size_t elements, T_INTERP_MODE mode, int batch = 1);

/**
 * \brief Evaluates the Boolean n > value for every input element n; outputs 1 if true, 0 if false
 * \param[in] d_input	Array with input numbers
 * \param[in] d_output	Array that will contain the result; d_output == d_input is valid if same type
 * \param[in] elements	Number of elements
 * \param[in] value	The value to be compared to input
 */
template<class T> void d_BiggerThan(tfloat* d_input, T* d_output, size_t elements, tfloat value);

/**
 * \brief Evaluates the Boolean n < value for every input element n; outputs 1 if true, 0 if false
 * \param[in] d_input	Array with input numbers
 * \param[in] d_output	Array that will contain the result; d_output == d_input is valid if same type
 * \param[in] elements	Number of elements
 * \param[in] value	The value to be compared to input
 */
template<class T> void d_SmallerThan(tfloat* d_input, T* d_output, size_t elements, tfloat value);

/**
 * \brief Evaluates the Boolean (n >= minval && n < maxval) for every input element n; outputs 1 if true, 0 if false
 * \param[in] d_input	Array with input numbers
 * \param[in] d_output	Array that will contain the result; d_output == d_input is valid if same type
 * \param[in] elements	Number of elements
 * \param[in] minval	The lower bound for the comparison
 * \param[in] maxval	The upper bound for the comparison
 */
template<class T> void d_IsBetween(tfloat* d_input, T* d_output, size_t elements, tfloat minval, tfloat maxval);

//MakeAtlas.cu:

/**
 * \brief Composes small images into one bigger, square image to reduce overhead from texture binding
 * \param[in] d_input	Array with image patches
 * \param[in] inputdims	X and Y are respective patch dimensions, Z is number of patches
 * \param[in] outputdims	Will contain the atlas dimensions
 * \param[in] primitivesperdim	X and Y will contain number of patches per atlas row and column, respectively
 * \param[in] h_primitivecoords	Host array that will contain the coordinates of the upper left corners of patches in atlas
 * \returns Atlas array; make sure to dispose manually
 */
template <class T> T* d_MakeAtlas(T* d_input, int3 inputdims, int3 &outputdims, int2 &primitivesperdim, int2* h_primitivecoords);

//Sum.cu:

/**
 * \brief Performs vector reduction by summing up the elements; use this version for few, but large vectors
 * \param[in] d_input	Array with input numbers
 * \param[in] d_output	Array that will contain the sum; d_output == d_input is not valid
 * \param[in] n	Number of elements
 * \param[in] batch	Number of vectors to be reduced
 */
template <class T> void d_Sum(T *d_input, T *d_output, size_t n, int batch = 1);

/**
 * \brief Performs vector reduction by summing up the elements; use this version for many, but small vectors (one CUDA block is used per vector)
 * \param[in] d_input	Array with input numbers
 * \param[in] d_output	Array that will contain the sum; d_output == d_input is not valid
 * \param[in] n	Number of elements
 * \param[in] batch	Number of vectors to be reduced
 */
template <class T> void d_SumMonolithic(T* d_input, T* d_output, int n, int batch);

//MinMax.cu:

/**
 * \brief Finds the smallest element and its position in a vector; use this version for few, but large vectors
 * \param[in] d_input	Array with input numbers
 * \param[in] d_output	Array that will contain the minimum value and its position
 * \param[in] n	Number of elements
 * \param[in] batch	Number of vectors
 */
template <class T> void d_Min(T *d_input, tuple2<T, size_t> *d_output, size_t n, int batch = 1);

/**
 * \brief Finds the smallest element in a vector; use this version for few, but large vectors
 * \param[in] d_input	Array with input numbers
 * \param[in] d_output	Array that will contain the minimum value
 * \param[in] n	Number of elements
 * \param[in] batch	Number of vectors
 */
template <class T> void d_Min(T *d_input, T *d_output, size_t n, int batch = 1);

/**
 * \brief Finds the largest element and its position in a vector; use this version for few, but large vectors
 * \param[in] d_input	Array with input numbers
 * \param[in] d_output	Array that will contain the maximum value and its position
 * \param[in] n	Number of elements
 * \param[in] batch	Number of vectors
 */
template <class T> void d_Max(T *d_input, tuple2<T, size_t> *d_output, size_t n, int batch = 1);

/**
 * \brief Finds the largest element in a vector; use this version for few, but large vectors
 * \param[in] d_input	Array with input numbers
 * \param[in] d_output	Array that will contain the maximum value
 * \param[in] n	Number of elements
 * \param[in] batch	Number of vectors
 */
template <class T> void d_Max(T *d_input, T *d_output, size_t n, int batch = 1);

//MinMaxMonolithic.cu:

/**
 * \brief Finds the smallest element and its position in a vector; use this version for many, but small vectors (one CUDA block is used per vector)
 * \param[in] d_input	Array with input numbers
 * \param[in] d_output	Array that will contain the minimum value and its position
 * \param[in] n	Number of elements
 * \param[in] batch	Number of vectors
 */
template <class T> void d_MinMonolithic(T* d_input, tuple2<T, size_t>* d_output, int n, int batch);

/**
 * \brief Finds the smallest element in a vector; use this version for many, but small vectors (one CUDA block is used per vector)
 * \param[in] d_input	Array with input numbers
 * \param[in] d_output	Array that will contain the minimum value
 * \param[in] n	Number of elements
 * \param[in] batch	Number of vectors
 */
template <class T> void d_MinMonolithic(T* d_input, T* d_output, int n, int batch);

/**
 * \brief Finds the largest element and its position in a vector; use this version for many, but small vectors (one CUDA block is used per vector)
 * \param[in] d_input	Array with input numbers
 * \param[in] d_output	Array that will contain the maximum value and its position
 * \param[in] n	Number of elements
 * \param[in] batch	Number of vectors
 */
template <class T> void d_MaxMonolithic(T* d_input, tuple2<T, size_t>* d_output, int n, int batch);

/**
 * \brief Finds the largest element in a vector; use this version for many, but small vectors (one CUDA block is used per vector)
 * \param[in] d_input	Array with input numbers
 * \param[in] d_output	Array that will contain the maximum value
 * \param[in] n	Number of elements
 * \param[in] batch	Number of vectors
 */
template <class T> void d_MaxMonolithic(T* d_input, T* d_output, int n, int batch);

//SumMinMax.cu:

/**
 * \brief Performs vector reduction by summing up the elements while also finding minimum and maximum values
 * \param[in] d_input	Array with input numbers
 * \param[in] d_sum	Array that will contain the sum; d_sum == d_input is not valid
 * \param[in] d_min	Array that will contain the minimum value; d_min == d_input is not valid
 * \param[in] d_max	Array that will contain the maximum value; d_max == d_input is not valid
 * \param[in] n	Number of elements
 * \param[in] batch	Number of vectors to be reduced
 */
template <class T> void d_SumMinMax(T* d_input, T* d_sum, T* d_min, T* d_max, size_t n, int batch = 1);

//Dev.cu:

/**
 * \brief Calculates the image metrics represented by a imgstats5 structure
 * \param[in] d_input	Array with input data
 * \param[in] d_output	Array that will contain the calculated metrics
 * \param[in] elements	Number of elements
 * \param[in] d_mask	Binary mask to restrict analysis to certain areas; optional: pass NULL to consider entire image
 * \param[in] batch	Number of images to be analyzed
 */
template <class Tmask> void d_Dev(tfloat* d_input, imgstats5* d_output, size_t elements, Tmask* d_mask, int batch = 1);

//Extraction.cu:

/**
 * \brief Extracts a rectangular portion from one or multiple images/volumes at a fixed location
 * \param[in] d_input	Array with input data
 * \param[in] d_output	Array that will contain the extracted images/volumes; d_output == d_input is not valid
 * \param[in] sourcedims	Dimensions of original image/volume
 * \param[in] regiondims	Dimensions of extracted portion
 * \param[in] regioncenter	Coordinates of the center point of the extracted portion; center is defined as dims / 2
 * \param[in] batch	Number of images to be processed
 */
template <class T> void d_Extract(T* d_input, T* d_output, int3 sourcedims, int3 regiondims, int3 regioncenter, int batch = 1);

/**
 * \brief Extracts rectangular portions from the same image/volume at different positions
 * \param[in] d_input	Array with input data
 * \param[in] d_output	Array that will contain the extracted images/volumes; d_output == d_input is not valid
 * \param[in] sourcedims	Dimensions of original image/volume
 * \param[in] regiondims	Dimensions of extracted portion
 * \param[in] d_regionorigins	Coordinates of the upper left corner of the extracted portion
 * \param[in] batch	Number of images to be extracted
 */
template <class T> void d_Extract(T* d_input, T* d_output, int3 sourcedims, int3 regiondims, int3* d_regionorigins, int batch = 1);

/**
 * \brief Extracts rectangular portions from the same image/volume at different positions, sampling it in a transformed reference frame
 * \param[in] d_input	Array with input data
 * \param[in] d_output	Array that will contain the extracted images/volumes; d_output == d_input is not valid
 * \param[in] sourcedims	Dimensions of original image/volume
 * \param[in] regiondims	Dimensions of extracted portion
 * \param[in] h_scale	Host array containing the scale factors
 * \param[in] h_rotation	Host array containing the rotation angles in radians
 * \param[in] h_translation	Host array containing the shift vectors; no shift = extracted center is at source's upper left corner
 * \param[in] mode	Interpolation mode; only T_INTERP_LINEAR and T_INTERP_CUBIC are supported
 * \param[in] batch	Number of images to be extracted
 */
void d_Extract2DTransformed(tfloat* d_input, tfloat* d_output, int3 sourcedims, int3 regiondims, tfloat2* h_scale, tfloat* h_rotation, tfloat2* h_translation, T_INTERP_MODE mode, int batch = 1);

//Padding.cu:
enum T_PAD_MODE 
{ 
	T_PAD_VALUE = 1,
	T_PAD_MIRROR = 2,
	T_PAD_TILE = 3
};
template <class T> void d_Pad(T* d_input, T* d_output, int3 inputdims, int3 outputdims, T_PAD_MODE mode, T value, int batch = 1);

//Reductions.cu:
template<class T> void d_ReduceAdd(T* d_input, T* d_output, int vectorlength, int nvectors, int batch = 1);


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
void d_Align2D(tfloat* d_input, tfloat* d_targets, int3 dims, int numtargets, tfloat3* h_params, int* h_membership, tfloat* h_scores, int maxtranslation, tfloat maxrotation, int iterations, T_ALIGN_MODE mode, int batch);

//Align3D.cu:
void d_Align3D(tfloat* d_input, tfloat* d_targets, int3 dims, int numtargets, tfloat3 &position, tfloat3 &rotation, int* h_membership, tfloat* h_scores, tfloat3* h_allpositions, tfloat3* h_allrotations, int maxtranslation, tfloat3 maxrotation, tfloat rotationstep, int rotationrefinements, T_ALIGN_MODE mode);


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
void d_Peak(tfloat* d_input, tfloat3* d_positions, tfloat* d_values, int3 dims, T_PEAK_MODE mode, cufftHandle* planforw = (cufftHandle*)NULL, cufftHandle* planback = (cufftHandle*)NULL, int batch = 1);
void d_PeakMakePlans(int3 dims, cufftHandle* planforw, cufftHandle* planback);
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
void d_IFFTC2R(tcomplex* const d_input, tfloat* const d_output, cufftHandle* plan);
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
void d_HermitianSymmetryMirrorHalf(tcomplex* d_input, tcomplex* d_output, int3 dims, int batch = 1);

//FFTRemap.cu:
template <class T> void d_RemapFull2HalfFFT(T* d_input, T* d_output, int3 dims, int batch = 1);
template <class T> void d_RemapFullFFT2Full(T* d_input, T* d_output, int3 dims, int batch = 1);
template <class T> void d_RemapFull2FullFFT(T* d_input, T* d_output, int3 dims, int batch = 1);
template <class T> void d_RemapHalfFFT2Half(T* d_input, T* d_output, int3 dims, int batch = 1);
template <class T> void d_RemapHalf2HalfFFT(T* d_input, T* d_output, int3 dims, int batch = 1);

//FFTResize.cu:
void d_FFTCrop(tcomplex* d_input, tcomplex* d_output, int3 olddims, int3 newdims, int batch = 1);
void d_FFTFullCrop(tcomplex* d_input, tcomplex* d_output, int3 olddims, int3 newdims, int batch = 1);
void d_FFTPad(tcomplex* d_input, tcomplex* d_output, int3 olddims, int3 newdims, int batch = 1);
void d_FFTFullPad(tcomplex* d_input, tcomplex* d_output, int3 olddims, int3 newdims, int batch = 1);

//////////////////////
//Image Manipulation//
//////////////////////

//AnisotropicLowpass:
void d_AnisotropicLowpass(tfloat* d_input, tfloat* d_output, int3 dims, tfloat* d_radiusmap, int2 anglesteps, tfloat smooth, cufftHandle* planforw, cufftHandle* planback, int batch);

//Bandpass.cu:
void d_Bandpass(tfloat* d_input, tfloat* d_output, int3 dims, tfloat low, tfloat high, tfloat smooth, tfloat* d_mask = NULL, cufftHandle* planforw = NULL, cufftHandle* planback = NULL, int batch = 1);
void d_BandpassNeat(tfloat* d_input, tfloat* d_output, int3 dims, tfloat low, tfloat high, tfloat smooth, int batch = 1);

//LocalLowpass.cu:
void d_LocalLowpass(tfloat* d_input, tfloat* d_output, int3 dims, tfloat* d_resolution, tfloat maxprecision);

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

//Xray.cu:
void d_Xray(tfloat* d_input, tfloat* d_output, int3 dims, tfloat ndev = (tfloat)4.6, int region = 2, int batch = 1);


///////////
//Masking//
///////////

//SphereMask.cu:
template <class T> void d_SphereMask(T* d_input, T* d_output, int3 size, tfloat* radius, tfloat sigma, tfloat3* center, int batch = 1);

//IrregularSphereMask.cu:
template <class T> void d_IrregularSphereMask(T* d_input, T* d_output, int3 dims, tfloat* radiusmap, int2 anglesteps, tfloat sigma, tfloat3* center, int batch = 1);

//RectangleMask.cu:
template <class T> void d_RectangleMask(T* d_input, T* d_output, int3 size, int3 rectsize, tfloat sigma, int3* center, int batch);

//Remap.cu:
template <class T> void d_Remap(T* d_input, intptr_t* d_map, T* d_output, size_t elementsmapped, size_t elementsoriginal, T defvalue, int batch = 1);
template <class T> void d_RemapReverse(T* d_input, intptr_t* d_map, T* d_output, size_t elementsmapped, size_t elementsdestination, T defvalue, int batch = 1);
template <class T> void Remap(T* h_input, intptr_t* h_map, T* h_output, size_t elementsmapped, size_t elementsoriginal, T defvalue, int batch = 1);
template <class T> void d_MaskSparseToDense(T* d_input, intptr_t** d_mapforward, intptr_t* d_mapbackward, size_t &elementsmapped, size_t elementsoriginal);
template <class T> void MaskSparseToDense(T* h_input, intptr_t** h_mapforward, intptr_t* h_mapbackward, size_t &elementsmapped, size_t elementsoriginal);

//Windows.cu:
void d_HannMask(tfloat* d_input, tfloat* d_output, int3 dims, tfloat* radius, tfloat3* center, int batch = 1);
void d_HammingMask(tfloat* d_input, tfloat* d_output, int3 dims, tfloat* radius, tfloat3* center, int batch = 1);
void d_GaussianMask(tfloat* d_input, tfloat* d_output, int3 dims, tfloat* sigma, tfloat3* center, int batch = 1);


//////////////
//Projection//
//////////////

//Backward.cu:
void d_ProjBackward(tfloat* d_volume, int3 dimsvolume, tfloat* d_image, int3 dimsimage, tfloat2* h_angles, tfloat* h_weights, T_INTERP_MODE mode, int batch = 1);

//Forward.cu:
void d_ProjForward(tfloat* d_volume, int3 dimsvolume, tfloat* d_image, tfloat* d_samples, int3 dimsimage, tfloat2* h_angles, int batch = 1);
void d_ProjForward2(tfloat* d_volume, int3 dimsvolume, tfloat* d_image, tfloat* d_samples, int3 dimsimage, tfloat2* h_angles, int batch = 1);


//////////////////
//Reconstruction//
//////////////////

//ART.cu:
void d_ART(tfloat* d_projections, int3 dimsproj, char* d_masks, tfloat* d_volume, tfloat* d_volumeerrors, int3 dimsvolume, tfloat2* h_angles, int iterations);

//RecFourier.cu:
void d_ReconstructFourier(tfloat* d_projections, int3 dimsproj, tfloat* d_volume, int3 dimsvolume, tfloat2* h_angles);


//////////////
//Resolution//
//////////////

enum T_FSC_MODE 
{ 
	T_FSC_THRESHOLD = 0,
	T_FSC_FIRSTMIN = 1
};

//FSC.cu:
void d_FSC(tfloat* d_volume1, tfloat* d_volume2, int3 dimsvolume, tfloat* d_curve, int maxradius, cufftHandle* plan = NULL, int batch = 1);

//LocalFSC.cu:
void d_LocalFSC(tfloat* d_volume1, tfloat* d_volume2, int3 dimsvolume, tfloat* d_resolution, int windowsize, int maxradius, tfloat threshold);

//AnisotropicFSC:
void d_AnisotropicFSC(tcomplex* d_volumeft1, tcomplex* d_volumeft2, int3 dimsvolume, tfloat* d_curve, int maxradius, tfloat3 direction, tfloat coneangle, tfloat falloff, int batch = 1);
void d_AnisotropicFSCMap(tfloat* d_volume1, tfloat* d_volume2, int3 dimsvolume, tfloat* d_map, int2 anglesteps, int maxradius, T_FSC_MODE fscmode, tfloat threshold, cufftHandle* plan, int batch);

//LocalAnisotropicFSC>
void d_LocalAnisotropicFSC(tfloat* d_volume1, tfloat* d_volume2, int3 dimsvolume, tfloat* d_resolution, int windowsize, int maxradius, int2 anglesteps, tfloat threshold);


//////////////
//Tomography//
//////////////

void d_InterpolateSingleAxisTilt(tcomplex* d_projft, int3 dimsproj, tcomplex* d_interpolated, tfloat* h_angles, int interpindex, int maxpoints, tfloat smoothsigma);


//////////////////
//Transformation//
//////////////////

//Bin.cu:
void d_Bin(tfloat* d_input, tfloat* d_output, int3 dims, int bincount, int batch = 1);

//Coordinates.cu:
void d_Cart2Polar(tfloat* d_input, tfloat* d_output, int2 dims, T_INTERP_MODE interpolation, int batch = 1);
void d_CartAtlas2Polar(tfloat* d_input, tfloat* d_output, tfloat2* d_offsets, int2 atlasdims, int2 dims, T_INTERP_MODE interpolation, int batch);
int2 GetCart2PolarSize(int2 dims);

//Rotate.cu:
void d_Rotate3D(tfloat* d_input, tfloat* d_output, int3 dims, tfloat3* angles, T_INTERP_MODE mode, int batch = 1);
void d_Rotate3D(cudaArray* a_input, cudaChannelFormatDesc channelDesc, tfloat* d_output, int3 dims, tfloat3* angles, T_INTERP_MODE mode, int batch = 1);
void d_Rotate2DFT(tcomplex* d_input, tcomplex* d_output, int3 dims, tfloat angle, T_INTERP_MODE mode, int batch = 1);
void d_Rotate2D(tfloat* d_input, tfloat* d_output, int3 dims, tfloat angle, int batch = 1);

//Shift.cu:
void d_Shift(tfloat* d_input, tfloat* d_output, int3 dims, tfloat3* delta, cufftHandle* planforw = NULL, cufftHandle* planback = NULL, tcomplex* d_sharedintermediate = NULL, int batch = 1);
void d_Shift(tcomplex* d_input, tcomplex* d_output, int3 dims, tfloat3* delta, bool iszerocentered = false, int batch = 1);

//Scale.cu:
void d_Scale(tfloat* d_input, tfloat* d_output, int3 olddims, int3 newdims, T_INTERP_MODE mode, cufftHandle* planforw = NULL, cufftHandle* planback = NULL, int batch = 1);
