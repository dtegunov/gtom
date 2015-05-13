#include "..\Prerequisites.h"


void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
    char const * const errId = "GTOM:Reconstruction:RecFourier:InvalidInput";
    char const * const errMsg = "Invalid input to MEX file.";

    mxInitGPU();

	if (nrhs < 4)
        mexErrMsgIdAndTxt(errId, "Wrong parameter count (4 expected).");

	mxArrayAdapter proj(prhs[0]);
	int3 dimsproj = MWDimsToInt3(mxGetNumberOfDimensions(proj.underlyingarray), mxGetDimensions(proj.underlyingarray));
	int3 dimsvolume = toInt3(dimsproj.x, dimsproj.x, dimsproj.x);
	int nimages = dimsproj.z;
	dimsproj.z = 1;
	tfloat* d_proj = proj.GetAsManagedDeviceTFloat();
	d_RemapFull2FullFFT(d_proj, d_proj, dimsproj, nimages);

	mxArrayAdapter weights(prhs[1]);
	int3 dimsweights = MWDimsToInt3(mxGetNumberOfDimensions(weights.underlyingarray), mxGetDimensions(weights.underlyingarray));
	tfloat* d_weights = NULL;
	if (ElementsFFT(dimsweights) > 1)
		d_weights = weights.GetAsManagedDeviceTFloat();

	mxArrayAdapter angles(prhs[2]);
	int3 dimsangles = MWDimsToInt3(mxGetNumberOfDimensions(angles.underlyingarray), mxGetDimensions(angles.underlyingarray));
	if (dimsangles.x != 3)
		mexErrMsgIdAndTxt(errId, "Angles must be a 3 x n sized matrix.");
	tfloat3* h_angles = (tfloat3*)angles.GetAsManagedTFloat();

	mxArrayAdapter shifts(prhs[3]);
	int3 dimsshifts = MWDimsToInt3(mxGetNumberOfDimensions(shifts.underlyingarray), mxGetDimensions(shifts.underlyingarray));
	if (dimsshifts.x != 2)
		mexErrMsgIdAndTxt(errId, "Shifts must be a 2 x n sized matrix.");
	tfloat2* h_shifts = (tfloat2*)shifts.GetAsManagedTFloat();

	tfloat* d_volume = CudaMallocValueFilled(Elements(dimsvolume), (tfloat)0);
	tfloat* d_volumepsf = CudaMallocValueFilled(ElementsFFT(dimsvolume), (tfloat)0);

	d_ReconstructFourierPrecise(d_proj, d_weights, d_volume, d_volumepsf, dimsvolume, h_angles, h_shifts, nimages, false);
	
	mwSize outputdims[3];
	outputdims[0] = dimsvolume.x;
	outputdims[1] = dimsvolume.y;
	outputdims[2] = dimsvolume.z;
	mxArrayAdapter A(mxCreateNumericArray(3,
					 outputdims,
					 mxGetClassID(proj.underlyingarray),
					 mxREAL));
	A.SetFromDeviceTFloat(d_volume);
	cudaFree(d_volume);
	plhs[0] = A.underlyingarray;

	outputdims[0] = dimsvolume.x / 2 + 1;
	outputdims[1] = dimsvolume.y;
	outputdims[2] = dimsvolume.z;
	mxArrayAdapter B(mxCreateNumericArray(3,
		outputdims,
		mxGetClassID(proj.underlyingarray),
		mxREAL));
	B.SetFromDeviceTFloat(d_volumepsf);
	cudaFree(d_volumepsf);
	plhs[1] = B.underlyingarray;

}