#include "..\Prerequisites.h"


void mexFunction(int nlhs, mxArray *plhs[],
	int nrhs, mxArray const *prhs[])
{
	char const * const errId = "GTOM:ImageManipulation:Bandpass:InvalidInput";

	mxInitGPU();

	if (nrhs != 3)
		mexErrMsgIdAndTxt(errId, "Wrong parameter count (3 expected).");

	mxArrayAdapter image(prhs[0]);
	int3 dimsimage = MWDimsToInt3(mxGetNumberOfDimensions(image.underlyingarray), mxGetDimensions(image.underlyingarray));
	tfloat* d_image = image.GetAsManagedDeviceTFloat();

	mxArrayAdapter lowfreq(prhs[1]);
	tfloat* h_lowfreq = lowfreq.GetAsManagedTFloat();

	mxArrayAdapter highfreq(prhs[2]);
	tfloat* h_highfreq = highfreq.GetAsManagedTFloat();

	d_Bandpass(d_image, d_image, dimsimage, h_lowfreq[0], h_highfreq[0], 0);

	mwSize outputdims[3];
	outputdims[0] = dimsimage.x;
	outputdims[1] = dimsimage.y;
	outputdims[2] = dimsimage.z;
	mxArrayAdapter A(mxCreateNumericArray(3,
		outputdims,
		mxGetClassID(image.underlyingarray),
		mxREAL));
	A.SetFromDeviceTFloat(d_image);
	plhs[0] = A.underlyingarray;
}