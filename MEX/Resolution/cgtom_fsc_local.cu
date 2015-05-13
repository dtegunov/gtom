#include "..\Prerequisites.h"


void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
    char const * const errId = "GTOM:Resolution:LocalFSC:InvalidInput";
    char const * const errMsg = "Invalid input to MEX file.";

    mxInitGPU();

	if (nrhs < 5)
        mexErrMsgIdAndTxt(errId, errMsg);

	mxArrayAdapter v1(prhs[0]);
	mxArrayAdapter v2(prhs[1]);
	int ndims = mxGetNumberOfDimensions(v1.underlyingarray);
	int3 dimsvolume = MWDimsToInt3(ndims, mxGetDimensions(v1.underlyingarray));
	uint nvolumes = dimsvolume.z / dimsvolume.x;
	dimsvolume.z = dimsvolume.x;

	tfloat* d_volume1 = v1.GetAsManagedDeviceTFloat();
	tfloat* d_volume2 = v2.GetAsManagedDeviceTFloat();
	tfloat* d_resmap = CudaMallocValueFilled(Elements(dimsvolume), (tfloat)0);

	d_LocalFSC(d_volume1, 
			   d_volume2, 
			   dimsvolume,
			   nvolumes,
			   d_resmap, 
			   ((double*)mxGetData(prhs[2]))[0], 
			   ((double*)mxGetData(prhs[3]))[0],
			   ((double*)mxGetData(prhs[4]))[0]);
	
	mxArrayAdapter B(mxCreateNumericArray(ndims,
					 mxGetDimensions(v1.underlyingarray),
					 mxGetClassID(v1.underlyingarray),
					 mxREAL));
	B.SetFromDeviceTFloat(d_resmap);
	cudaFree(d_resmap);
	plhs[0] = B.underlyingarray;
}