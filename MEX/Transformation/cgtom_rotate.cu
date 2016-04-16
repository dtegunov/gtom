#include "..\Prerequisites.h"


void mexFunction(int nlhs, mxArray *plhs[],
	int nrhs, mxArray const *prhs[])
{
	char const * const errId = "GTOM:Transformation:Rotate:InvalidInput";

	mxInitGPU();

	if (nrhs != 2)
		mexErrMsgIdAndTxt(errId, "Not enough parameters (2 expected).");

	mxArrayAdapter images(prhs[0]);
	int3 dimsimage = MWDimsToInt3(mxGetNumberOfDimensions(images.underlyingarray), mxGetDimensions(images.underlyingarray));
	tfloat* d_image = images.GetAsManagedDeviceTFloat();

	mxArrayAdapter angles(prhs[1]);
	tfloat* h_angles = angles.GetAsManagedTFloat();

	tfloat* d_transformed;
	cudaMalloc((void**)&d_transformed, Elements(dimsimage) * sizeof(tfloat));

	if (DimensionCount(dimsimage) == 3)
		d_Rotate3D(d_image, d_transformed, dimsimage, (tfloat3*)h_angles, 1, T_INTERP_CUBIC, true);
	else
		d_Rotate2D(d_image, d_transformed, toInt2(dimsimage), h_angles, T_INTERP_CUBIC, true);

	mwSize outputdims[3];
	outputdims[0] = dimsimage.x;
	outputdims[1] = dimsimage.y;
	outputdims[2] = dimsimage.z;
	mxArrayAdapter A(mxCreateNumericArray(3,
		outputdims,
		mxGetClassID(images.underlyingarray),
		mxREAL));
	A.SetFromDeviceTFloat(d_transformed);
	plhs[0] = A.underlyingarray;

	cudaFree(d_transformed);
}
