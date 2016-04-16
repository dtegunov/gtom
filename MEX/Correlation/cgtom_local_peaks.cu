#include "..\Prerequisites.h"


void mexFunction(int nlhs, mxArray *plhs[],
	int nrhs, mxArray const *prhs[])
{
	char const * const errId = "GTOM:Correlation:LocalPeaks:InvalidInput";

	mxInitGPU();

	if (nrhs != 3)
		mexErrMsgIdAndTxt(errId, "Wrong parameter count (3 expected).");

	mxArrayAdapter image(prhs[0]);
	int3 dimsimage = MWDimsToInt3(mxGetNumberOfDimensions(image.underlyingarray), mxGetDimensions(image.underlyingarray));
	tfloat* d_image = image.GetAsManagedDeviceTFloat();

	int ndims = DimensionCount(dimsimage);

	mxArrayAdapter threshold(prhs[1]);
	tfloat* h_threshold = threshold.GetAsManagedTFloat();

	mxArrayAdapter extent(prhs[2]);
	tfloat* h_extent = extent.GetAsManagedTFloat();

	int3** h_peaks = (int3**)malloc(sizeof(int3*));
	int* h_peakcount = (int*)malloc(sizeof(int));

	d_LocalPeaks(d_image, h_peaks, h_peakcount, dimsimage, (int)(h_extent[0] + 0.1f), h_threshold[0]);

	mwSize outputdims[3];
	outputdims[0] = 3;
	outputdims[1] = h_peakcount[0];
	outputdims[2] = 1;
	mxArrayAdapter A(mxCreateNumericArray(3,
		outputdims,
		mxINT32_CLASS,
		mxREAL));
	A.SetFromTFloat((tfloat*)h_peaks[0]);
	plhs[0] = A.underlyingarray;


	free(h_peaks[0]);
	free(h_peaks);
	free(h_peakcount);
}