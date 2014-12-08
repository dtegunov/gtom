#include "..\Prerequisites.h"


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
	if (ndims < 1 || ndims > 3)
		mexErrMsgIdAndTxt(errId, errMsg);
	int3 dimensions = MWDimsToInt3(ndims, mxGetDimensions(A.underlyingarray));
	tcomplex* d_result;
	cudaMalloc((void**)&d_result, dimensions.x * dimensions.y * dimensions.z * sizeof(tcomplex));

    d_FFTC2C(A.GetAsManagedDeviceTComplex(), d_result, ndims, dimensions);

	mxArrayAdapter B(mxCreateNumericArray(mxGetNumberOfDimensions(A.underlyingarray),
					 mxGetDimensions(A.underlyingarray),
					 mxGetClassID(A.underlyingarray),
					 mxCOMPLEX));
	B.SetFromDeviceTComplex(d_result);
	plhs[0] = B.underlyingarray;

	cudaFree(d_result);
}