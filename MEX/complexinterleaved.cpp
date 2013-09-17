#include "Prerequisites.h"


void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
    char const * const errId = "parallel:gpu:mexGPUExample:InvalidInput";
    char const * const errMsg = "Invalid input to MEX file.";

    mxInitGPU();

	if (nrhs != 1 || !mxIsComplex(prhs[0]))
        mexErrMsgIdAndTxt(errId, errMsg);

	mxArrayAdapter A(prhs[0]);
	int ndims = mxGetNumberOfDimensions(A.underlyingarray);
	int3 dimensions = MWDimsToInt3(ndims, mxGetDimensions(A.underlyingarray));

	mwSize realdims[3] = { dimensions.x * 2, dimensions.y, dimensions.z };
	mxArrayAdapter B(mxCreateNumericArray(mxGetNumberOfDimensions(A.underlyingarray),
										  realdims,
										  mxGetClassID(A.underlyingarray),
										  mxREAL));
	B.SetFromTFloat((tfloat*)A.GetAsManagedTComplex());
	plhs[0] = B.underlyingarray;
}