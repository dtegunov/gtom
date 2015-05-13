#include "..\Prerequisites.h"


void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
    char const * const errId = "GTOM:Reconstruction:SIRT:InvalidInput";
    char const * const errMsg = "Wrong number of arguments (6 expected).";

    mxInitGPU();

	if (nrhs < 6)
        mexErrMsgIdAndTxt(errId, errMsg);

	mxArrayAdapter proj(prhs[0]);
	int ndims = mxGetNumberOfDimensions(proj.underlyingarray);
	int3 dimsproj = MWDimsToInt3(ndims, mxGetDimensions(proj.underlyingarray));
	int nimages = dimsproj.z;
	tfloat* d_proj = proj.GetAsManagedDeviceTFloat();
	
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

	mxArrayAdapter scales(prhs[4]);
	int3 dimsscales = MWDimsToInt3(mxGetNumberOfDimensions(scales.underlyingarray), mxGetDimensions(scales.underlyingarray));
	if (dimsscales.x != 2)
		mexErrMsgIdAndTxt(errId, "Scales must be a 2 x n sized matrix.");
	tfloat2* h_scales = (tfloat2*)scales.GetAsManagedTFloat();

	tfloat2* h_intensities = (tfloat2*)malloc(nimages * sizeof(tfloat2));
	for (int i = 0; i < nimages; i++)
		h_intensities[i] = tfloat2(0, 1);

	int iterations = (int)((double*)mxGetData(prhs[5]))[0];

	int3 dimsvolume = toInt3((int)((double*)mxGetData(prhs[1]))[0], (int)((double*)mxGetData(prhs[1]))[1], (int)((double*)mxGetData(prhs[1]))[2]);
	tfloat* d_volume = CudaMallocValueFilled(Elements(dimsvolume), (tfloat)0);

	d_RecSIRT(d_volume, NULL, dimsvolume, tfloat3(0), d_proj, toInt2(dimsproj), nimages, h_angles, h_shifts, h_scales, h_intensities, T_INTERP_CUBIC, 1, iterations, true);
	
	free(h_intensities);

	mwSize outputdims[3];
	outputdims[0] = dimsvolume.x;
	outputdims[1] = dimsvolume.y;
	outputdims[2] = dimsvolume.z;
	mxArrayAdapter B(mxCreateNumericArray(3,
					 outputdims,
					 mxGetClassID(proj.underlyingarray),
					 mxREAL));
	B.SetFromDeviceTFloat(d_volume);
	plhs[0] = B.underlyingarray;
	cudaFree(d_volume);
}