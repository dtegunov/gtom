#include "..\Prerequisites.h"


void mexFunction(int nlhs, mxArray *plhs[],
	int nrhs, mxArray const *prhs[])
{
	char const * const errId = "GTOM:Projection:Weighting:InvalidInput";

	mxInitGPU();

	if (nrhs < 2)
		mexErrMsgIdAndTxt(errId, "Not enough parameters (2 or 3 expected).");

	mxArrayAdapter imagesize(prhs[0]);
	tfloat* h_imagesize = imagesize.GetAsManagedTFloat();
	int2 dimsimage = toInt2((int)(h_imagesize[0] + 0.5f), (int)(h_imagesize[1] + 0.5f));

	mxArrayAdapter angles(prhs[1]);
	int ndims = mxGetNumberOfDimensions(angles.underlyingarray);
	int3 dimsangles = MWDimsToInt3(ndims, mxGetDimensions(angles.underlyingarray));
	if (dimsangles.x != 3)
		mexErrMsgIdAndTxt(errId, "3 values per column expected for angles.");
	tfloat3* h_angles = (tfloat3*)angles.GetAsManagedTFloat();

	int nimages = dimsangles.y;

	tfloat* d_weights;
	cudaMalloc((void**)&d_weights, (dimsimage.x / 2 + 1) * dimsimage.y * nimages * sizeof(tfloat));

	d_ExactWeighting(d_weights, dimsimage, h_angles, nimages, dimsimage.x / 2, true);

	mwSize outputdims[3];
	outputdims[0] = dimsimage.x / 2 + 1;
	outputdims[1] = dimsimage.y;
	outputdims[2] = nimages;
	mxArrayAdapter A(mxCreateNumericArray(3,
		outputdims,
		mxDOUBLE_CLASS,
		mxREAL));
	A.SetFromDeviceTFloat(d_weights);
	cudaFree(d_weights);
	plhs[0] = A.underlyingarray;
}