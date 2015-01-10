#include "..\Prerequisites.h"


void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
    char const * const errId = "GTOM:Reconstruction:RecFourier:InvalidInput";
    char const * const errMsg = "Invalid input to MEX file.";

    mxInitGPU();

	if (nrhs < 3)
        mexErrMsgIdAndTxt(errId, errMsg);

	mxArrayAdapter proj(prhs[0]);
	int ndims = mxGetNumberOfDimensions(proj.underlyingarray);
	int3 dimsproj = MWDimsToInt3(ndims, mxGetDimensions(proj.underlyingarray));
	tfloat* d_proj = proj.GetAsManagedDeviceTFloat();

	mxArrayAdapter angles(prhs[1]);
	ndims = mxGetNumberOfDimensions(angles.underlyingarray);
	int3 dimsangles = MWDimsToInt3(ndims, mxGetDimensions(angles.underlyingarray));
	tfloat2* h_angles = (tfloat2*)angles.GetAsManagedTFloat();

	int3 dimsvolume = toInt3((int)((double*)mxGetData(prhs[2]))[0], (int)((double*)mxGetData(prhs[2]))[1], (int)((double*)mxGetData(prhs[2]))[2]);
	tfloat* d_volume = CudaMallocValueFilled(Elements(dimsvolume), (tfloat)0);

	d_ReconstructFourier(d_proj, dimsproj, d_volume, dimsvolume, h_angles);
	
	mwSize outputdims[3];
	outputdims[0] = dimsvolume.x;
	outputdims[1] = dimsvolume.y;
	outputdims[2] = dimsvolume.z;
	mxArrayAdapter B(mxCreateNumericArray(3,
					 outputdims,
					 mxGetClassID(proj.underlyingarray),
					 mxREAL));
	B.SetFromDeviceTFloat(d_volume);
	cudaFree(d_volume);
	plhs[0] = B.underlyingarray;
}