#include "..\Prerequisites.h"


void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
    char const * const errId = "GTOM:Resolution:AnisotropicFSCMap:InvalidInput";
    char const * const errMsg = "Invalid input to MEX file.";

    mxInitGPU();

	if (nrhs < 5)
        mexErrMsgIdAndTxt(errId, errMsg);

	mxArrayAdapter v1(prhs[0]);
	mxArrayAdapter v2(prhs[1]);
	int ndims = mxGetNumberOfDimensions(v1.underlyingarray);
	int3 dimsvolume = MWDimsToInt3(ndims, mxGetDimensions(v1.underlyingarray));
	int2 anglesteps = toInt2((int)((double*)mxGetData(prhs[2]))[0], (int)((double*)mxGetData(prhs[2]))[1]);

	tfloat* d_volume1 = v1.GetAsManagedDeviceTFloat();
	tfloat* d_volume2 = v2.GetAsManagedDeviceTFloat();
	tfloat* d_resmap = CudaMallocValueFilled(anglesteps.x * anglesteps.y, (tfloat)0);

	d_AnisotropicFSCMap(d_volume1, 
						d_volume2, 
						dimsvolume, 
						d_resmap, 
						anglesteps, 
						((double*)mxGetData(prhs[3]))[0],
						((double*)mxGetData(prhs[4]))[0],
						NULL,
						1);

	mwSize mapsize[3];
	mapsize[0] = (mwSize)anglesteps.x;
	mapsize[1] = (mwSize)anglesteps.y;
	mapsize[2] = (mwSize)0;
	mxArrayAdapter B(mxCreateNumericArray(2,
					 mapsize,
					 mxGetClassID(v1.underlyingarray),
					 mxREAL));
	B.SetFromDeviceTFloat(d_resmap);
	cudaFree(d_resmap);
	plhs[0] = B.underlyingarray;
}