#include "..\Prerequisites.h"


void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
    char const * const errId = "parallel:gpu:mexGPUExample:InvalidInput";
    char const * const errMsg = "Invalid input to MEX file.";

    mxInitGPU();

	if (nrhs != 2 || !mxIsComplex(prhs[0]) || (int)mxGetDimensions(prhs[0])[0] != (int)mxGetPr(prhs[1])[0] / 2 + 1)
        mexErrMsgIdAndTxt(errId, errMsg);

	mxArrayAdapter A(prhs[0]);
	int ndims = mxGetNumberOfDimensions(A.underlyingarray);
	if (ndims < 1 || ndims > 3)
		mexErrMsgIdAndTxt(errId, errMsg);
	int3 dimensions = MWDimsToInt3(ndims, mxGetPr(prhs[1]));
	tfloat* d_result;
	cudaMalloc((void**)&d_result, dimensions.x * dimensions.y * dimensions.z * sizeof(tfloat));

    d_IFFTC2R(A.GetAsManagedDeviceTComplex(), d_result, ndims, dimensions);

	mwSize realdims[3] = { dimensions.x, dimensions.y, dimensions.z };
	mxArrayAdapter B(mxCreateNumericArray(mxGetNumberOfDimensions(A.underlyingarray),
										  realdims,
										  mxGetClassID(A.underlyingarray),
										  mxREAL));
	B.SetFromDeviceTFloat(d_result);
	plhs[0] = B.underlyingarray;

	cudaFree(d_result);
}