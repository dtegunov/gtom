#include "Prerequisites.cuh"

#ifndef BINARY_CUH
#define BINARY_CUH

namespace gtom
{
	///////////////////////
	//Binary manipulation//
	///////////////////////

	/**
	* \brief Performs a dilation operation on a binary image/volume
	* \param[in] d_input	Array with input data
	* \param[in] d_output	Array that will contain the dilated data
	* \param[in] dims	Array dimensions
	* \param[in] batch	Number of arrays
	*/
	template <class T> void d_Dilate(T* d_input, T* d_output, int3 dims, int batch = 1);

	/**
	* \brief Converts floating point data to binary by applying a threshold; value >= threshold is set to 1, otherwise 0; binary data type can be char or int
	* \param[in] d_input	Array with input data
	* \param[in] d_output	Array that will contain the binarized data
	* \param[in] elements	Number of elements in array
	* \param[in] threshold	Threshold to be applied
	* \param[in] batch	Number of arrays
	*/
	template <class T> void d_Binarize(tfloat* d_input, T* d_output, size_t elements, tfloat threshold, int batch = 1);
}
#endif