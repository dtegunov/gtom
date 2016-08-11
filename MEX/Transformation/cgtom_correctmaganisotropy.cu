#include "..\Prerequisites.h"


void mexFunction(int nlhs, mxArray *plhs[],
	int nrhs, mxArray const *prhs[])
{
	char const * const errId = "GTOM:Transformation:CorrectMagAnisotropy:InvalidInput";

	mxInitGPU();

	if (nrhs != 3)
		mexErrMsgIdAndTxt(errId, "Not enough parameters (3 expected).");

	mxArrayAdapter images(prhs[0]);
	int3 dimsimage = MWDimsToInt3(mxGetNumberOfDimensions(images.underlyingarray), mxGetDimensions(images.underlyingarray));
	tfloat* d_image = images.GetAsManagedDeviceTFloat();

	int nimages = dimsimage.z;
	dimsimage.z = 1;

	mxArrayAdapter anisoparams(prhs[1]);
	float* h_anisoparams = anisoparams.GetAsManagedTFloat();

	mxArrayAdapter superscale(prhs[2]);
	float* h_superscale = superscale.GetAsManagedTFloat();

	d_MagAnisotropyCorrect(d_image, toInt2(dimsimage), d_image, toInt2(dimsimage), h_anisoparams[0], h_anisoparams[1], h_anisoparams[2], (uint)h_superscale[0], nimages);

	mwSize outputdims[3];
	outputdims[0] = dimsimage.x;
	outputdims[1] = dimsimage.y;
	outputdims[2] = nimages;
	mxArrayAdapter A(mxCreateNumericArray(3,
		outputdims,
		mxGetClassID(images.underlyingarray),
		mxREAL));
	A.SetFromDeviceTFloat(d_image);
	plhs[0] = A.underlyingarray;
}
