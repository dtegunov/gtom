#include "cufft.h"
#include "Prerequisites.cuh"

#ifndef CORRELATION_CUH
#define CORRELATION_CUH

namespace gtom
{
	///////////////
	//Correlation//
	///////////////

	//CCF.cu:

	/**
	* \brief Computes the correlation surface by folding two maps in Fourier space; an FFTShift operation is applied afterwards, so that center = no translation
	* \param[in] d_input1	First input map
	* \param[in] d_input2	Second input map
	* \param[in] d_output	Array that will contain the correlation data
	* \param[in] dims	Array dimensions
	* \param[in] normalized	Indicates if the input maps are already normalized, i. e. if this step can be skipped
	* \param[in] d_mask	Optional mask used during normalization, if normalized = false
	* \param[in] batch	Number of map pairs
	*/
	template<class T> void d_CCF(tfloat* d_input1, tfloat* d_input2, tfloat* d_output, int3 dims, bool normalized, T* d_mask, int batch = 1);

	/**
	* \brief Computes the correlation surface by folding two maps in Fourier space; FFTShift operation is not applied afterwards, i. e. first element in array = no translation
	* \param[in] d_input1	First input map
	* \param[in] d_input2	Second input map
	* \param[in] d_output	Array that will contain the correlation data
	* \param[in] dims	Array dimensions
	* \param[in] normalized	Indicates if the input maps are already normalized, i. e. if this step can be skipped
	* \param[in] d_mask	Optional mask used during normalization, if normalized = false
	* \param[in] batch	Number of map pairs
	*/
	template<class T> void d_CCFUnshifted(tfloat* d_input1, tfloat* d_input2, tfloat* d_output, int3 dims, bool normalized, T* d_mask, int batch = 1);

	//Peak.cu:

	/**
	* \brief Specifies how the position of a peak should be determined
	*/
	enum T_PEAK_MODE
	{
		/**Only integer values; fastest*/
		T_PEAK_INTEGER = 1,
		/**Subpixel precision, but with x, y and z determined by scaling a row/column independently in each dimension; moderately fast*/
		T_PEAK_SUBCOARSE = 2,
		/**Subpixel precision, with a portion around the peak extracted and up-scaled in Fourier space; slow, but highest precision*/
		T_PEAK_SUBFINE = 3
	};

	/**
	* \brief Locates the position of the maximum value in a map with the specified precision
	* \param[in] d_input	Array with input data
	* \param[in] d_positions	Array that will contain the peak position for each map in batch
	* \param[in] d_values	Array that will contain the peak values for each map in batch
	* \param[in] dims	Array dimensions
	* \param[in] mode	Desired positional precision
	* \param[in] planforw	Optional pre-cooked forward FFT plan; can be made with d_PeakMakePlans
	* \param[in] planback	Optional pre-cooked reverse FFT plan; can be made with d_PeakMakePlans
	* \param[in] batch	Number of maps
	*/
	void d_Peak(tfloat* d_input, tfloat3* d_positions, tfloat* d_values, int3 dims, T_PEAK_MODE mode, cufftHandle* planforw = (cufftHandle*)NULL, cufftHandle* planback = (cufftHandle*)NULL, int batch = 1);


	/**
	* \brief Detects multiple local peaks in a map
	* \param[in] d_input	Array with input data
	* \param[in] h_peaks	Pointer that will contain a host array with peak positions
	* \param[in] h_peaksnum	Host array that will contain the number of peaks in each map
	* \param[in] localextent	Distance that a peak has to be apart from a higher/equal peak to be detected
	* \param[in] threshold	Minimum value for peaks to be considered
	* \param[in] batch	Number of maps
	*/
	void d_LocalPeaks(tfloat* d_input, int3** h_peaks, int* h_peaksnum, int3 dims, int localextent, tfloat threshold, int batch = 1);

	//SimilarityMatrix.cu:

	void d_RotationSeries(tfloat* d_image, tfloat* d_series, int2 dimsimage, int anglesteps);
	void d_SimilarityMatrixRow(tfloat* d_images, tcomplex* d_imagesft, int2 dimsimage, int nimages, int anglesteps, int target, tfloat* d_similarity);
	void d_LineSimilarityMatrixRow(tcomplex* d_linesft, int2 dimsimage, int nimages, int linewidth, int anglesteps, int target, tfloat* d_similarity);
}
#endif