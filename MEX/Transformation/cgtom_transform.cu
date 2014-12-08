#include "..\Prerequisites.h"


void mexFunction(int nlhs, mxArray *plhs[],
	int nrhs, mxArray const *prhs[])
{
	char const * const errId = "GTOM:Transformation:ScaleRotateShift:InvalidInput";

	mxInitGPU();

	if (nrhs < 2)
		mexErrMsgIdAndTxt(errId, "Not enough parameters (2 or 3 expected).");

	mxArrayAdapter images(prhs[0]);
	int3 dimsimages = MWDimsToInt3(mxGetNumberOfDimensions(images.underlyingarray), mxGetDimensions(images.underlyingarray));
	tfloat* d_images = images.GetAsManagedDeviceTFloat();

	mxArrayAdapter scales(prhs[1]);
	tfloat2* h_scales = (tfloat2*)scales.GetAsManagedTFloat();
	if (MWDimsToInt3(mxGetNumberOfDimensions(scales.underlyingarray), mxGetDimensions(scales.underlyingarray)).x != 2)
		mexErrMsgIdAndTxt(errId, "Scale must be given as 2 values (x and y) per column.");

	mxArrayAdapter angles(prhs[2]);
	tfloat* h_angles = angles.GetAsManagedTFloat();

	mxArrayAdapter shifts(prhs[3]);
	tfloat2* h_shifts = (tfloat2*)shifts.GetAsManagedTFloat();
	if (MWDimsToInt3(mxGetNumberOfDimensions(shifts.underlyingarray), mxGetDimensions(shifts.underlyingarray)).x != 2)
		mexErrMsgIdAndTxt(errId, "Shifts must be given as 2 values (x and y) per column.");

	int nimages = dimsimages.z;

	tfloat* d_transformed;
	cudaMalloc((void**)&d_transformed, Elements(dimsimages) * sizeof(tfloat));

	d_ScaleRotateShift2D(d_images, d_transformed, toInt2(dimsimages.x, dimsimages.y), h_scales, h_angles, h_shifts, T_INTERP_CUBIC, true, nimages);

	mwSize outputdims[3];
	outputdims[0] = dimsimages.x;
	outputdims[1] = dimsimages.y;
	outputdims[2] = dimsimages.z;
	mxArrayAdapter A(mxCreateNumericArray(3,
		outputdims,
		mxGetClassID(images.underlyingarray),
		mxREAL));
	A.SetFromDeviceTFloat(d_transformed);
	cudaFree(d_transformed);
	plhs[0] = A.underlyingarray;
}