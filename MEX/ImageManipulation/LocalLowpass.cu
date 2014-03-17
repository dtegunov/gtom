#include "..\Prerequisites.h"


void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
    char const * const errId = "GTOM:Resolution:LocalFSC:InvalidInput";
    char const * const errMsg = "Invalid input to MEX file.";

    mxInitGPU();

	if (nrhs < 3)
        mexErrMsgIdAndTxt(errId, errMsg);

	mxArrayAdapter vol(prhs[0]);
	int ndims = mxGetNumberOfDimensions(vol.underlyingarray);
	int3 dimsvolume = MWDimsToInt3(ndims, mxGetDimensions(vol.underlyingarray));
	mxArrayAdapter resmap(prhs[1]);

	tfloat* d_volume = vol.GetAsManagedDeviceTFloat();
	tfloat* d_resmap = resmap.GetAsManagedDeviceTFloat();
	tfloat* d_output = CudaMallocValueFilled(Elements(dimsvolume), (tfloat)0);

	d_LocalLowpass(d_volume, d_output, dimsvolume, d_resmap, (tfloat)((double*)mxGetData(prhs[2]))[0]);
	
	mxArrayAdapter B(mxCreateNumericArray(ndims,
					 mxGetDimensions(vol.underlyingarray),
					 mxGetClassID(vol.underlyingarray),
					 mxREAL));
	B.SetFromDeviceTFloat(d_output);
	cudaFree(d_output);
	plhs[0] = B.underlyingarray;
}