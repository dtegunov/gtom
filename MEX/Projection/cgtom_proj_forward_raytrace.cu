#include "..\Prerequisites.h"


void mexFunction(int nlhs, mxArray *plhs[],
	int nrhs, mxArray const *prhs[])
{
	char const * const errId = "GTOM:Projection:ForwardProjRaytrace:InvalidInput";

	mxInitGPU();

	if (nrhs != 4)
		mexErrMsgIdAndTxt(errId, "Wrong parameter count (4 expected).");

	mxArrayAdapter volume(prhs[0]);
	int ndims = mxGetNumberOfDimensions(volume.underlyingarray);
	int3 dimsvolume = MWDimsToInt3(ndims, mxGetDimensions(volume.underlyingarray));
	tfloat* d_volume = volume.GetAsManagedDeviceTFloat();

	mxArrayAdapter angles(prhs[1]);
	int3 dimsangles = MWDimsToInt3(mxGetNumberOfDimensions(angles.underlyingarray), mxGetDimensions(angles.underlyingarray));
	if (dimsangles.x != 3)
		mexErrMsgIdAndTxt(errId, "3 values per column expected for angles.");
	tfloat3* h_angles = (tfloat3*)angles.GetAsManagedTFloat();

	mxArrayAdapter shifts(prhs[2]);
	int3 dimsshifts = MWDimsToInt3(mxGetNumberOfDimensions(shifts.underlyingarray), mxGetDimensions(shifts.underlyingarray));
	if (dimsshifts.x != 2)
		mexErrMsgIdAndTxt(errId, "2 values per column expected for shifts.");
	tfloat2* h_shifts = (tfloat2*)shifts.GetAsManagedTFloat();

	mxArrayAdapter scales(prhs[3]);
	int3 dimsscales = MWDimsToInt3(mxGetNumberOfDimensions(scales.underlyingarray), mxGetDimensions(scales.underlyingarray));
	if (dimsscales.x != 2)
		mexErrMsgIdAndTxt(errId, "2 values per column expected for scales.");
	tfloat2* h_scales = (tfloat2*)scales.GetAsManagedTFloat();

	int2 dimsproj = toInt2(dimsvolume.x, dimsvolume.x);
	tfloat* d_proj;
	cudaMalloc((void**)&d_proj, Elements2(dimsproj) * dimsangles.y * sizeof(tfloat));

	d_ProjForwardRaytrace(d_volume, dimsvolume, d_proj, dimsproj, h_angles, h_shifts, h_scales, T_INTERP_CUBIC, 1, dimsangles.y);

	mwSize outputdims[3];
	outputdims[0] = dimsproj.x;
	outputdims[1] = dimsproj.y;
	outputdims[2] = dimsangles.y;
	mxArrayAdapter A(mxCreateNumericArray(3,
		outputdims,
		mxGetClassID(volume.underlyingarray),
		mxREAL));
	A.SetFromDeviceTFloat(d_proj);
	cudaFree(d_proj);
	plhs[0] = A.underlyingarray;
}