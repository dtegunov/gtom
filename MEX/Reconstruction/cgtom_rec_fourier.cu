#include "..\Prerequisites.h"


void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
    char const * const errId = "GTOM:Reconstruction:RecFourier:InvalidInput";
    char const * const errMsg = "Invalid input to MEX file.";

    mxInitGPU();

	if (nrhs < 5)
        mexErrMsgIdAndTxt(errId, "Wrong parameter count (4 expected).");

	mxArrayAdapter proj(prhs[0]);
	int ndims = mxGetNumberOfDimensions(proj.underlyingarray);
	int3 dimsproj = MWDimsToInt3(ndims, mxGetDimensions(proj.underlyingarray));
	tfloat* d_proj = proj.GetAsManagedDeviceTFloat();

	mxArrayAdapter weights(prhs[1]);
	int3 dimsweights = MWDimsToInt3(mxGetNumberOfDimensions(weights.underlyingarray), mxGetDimensions(weights.underlyingarray));
	tfloat* d_weights = NULL;
	if (ElementsFFT(dimsweights) > 1)
		d_weights = weights.GetAsManagedDeviceTFloat();

	mxArrayAdapter ctf(prhs[2]);
	int3 dimsctf = MWDimsToInt3(mxGetNumberOfDimensions(ctf.underlyingarray), mxGetDimensions(ctf.underlyingarray));
	CTFParams* h_ctf = NULL;
	if (ElementsFFT(dimsctf) > 1)
		h_ctf = (CTFParams*)ctf.GetAsManagedTFloat();

	mxArrayAdapter angles(prhs[3]);
	ndims = mxGetNumberOfDimensions(angles.underlyingarray);
	int3 dimsangles = MWDimsToInt3(ndims, mxGetDimensions(angles.underlyingarray));
	if (dimsangles.x != 3)
		mexErrMsgIdAndTxt(errId, "Angles must contain 3 values per column.");
	tfloat3* h_angles = (tfloat3*)angles.GetAsManagedTFloat();

	int3 dimsvolume = toInt3((int)(((double*)mxGetData(prhs[4]))[0] + 0.5), (int)(((double*)mxGetData(prhs[4]))[1] + 0.5), (int)(((double*)mxGetData(prhs[4]))[2] + 0.5));
	tfloat* d_volume = CudaMallocValueFilled(Elements(dimsvolume), (tfloat)0);

	d_ReconstructFourier(d_proj, d_weights, h_ctf, dimsproj, d_volume, dimsvolume, h_angles);
	
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