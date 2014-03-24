#include "..\Prerequisites.h"


void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
    char const * const errId = "GTOM:Resolution:AnisotropicLowpass:InvalidInput";
    char const * const errMsg = "Invalid input to MEX file.";

    mxInitGPU();

	if (nrhs < 4)
        mexErrMsgIdAndTxt(errId, errMsg);

	mxArrayAdapter vol(prhs[0]);
	int ndims = mxGetNumberOfDimensions(vol.underlyingarray);
	int3 dimsvolume = MWDimsToInt3(ndims, mxGetDimensions(vol.underlyingarray));
	mxArrayAdapter resmap(prhs[1]);
	int2 anglesteps = toInt2((int)((double*)mxGetData(prhs[2]))[0], (int)((double*)mxGetData(prhs[2]))[1]);

	tfloat* d_volume = vol.GetAsManagedDeviceTFloat();
	tfloat* d_resmap = resmap.GetAsManagedDeviceTFloat();
	tfloat* d_output = CudaMallocValueFilled(Elements(dimsvolume), (tfloat)0);

	d_AnisotropicLowpass(d_volume, d_output, dimsvolume, d_resmap, anglesteps, ((double*)mxGetData(prhs[3]))[0], NULL, NULL, 1);
	
	mxArrayAdapter B(mxCreateNumericArray(ndims,
					 mxGetDimensions(vol.underlyingarray),
					 mxGetClassID(vol.underlyingarray),
					 mxREAL));
	B.SetFromDeviceTFloat(d_output);
	cudaFree(d_output);
	plhs[0] = B.underlyingarray;
}