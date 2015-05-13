#include "..\Prerequisites.h"


void mexFunction(int nlhs, mxArray *plhs[],
	int nrhs, mxArray const *prhs[])
{
	char const * const errId = "GTOM:Projection:ForwardProjFourier:InvalidInput";

	mxInitGPU();

	if (nrhs != 4)
		mexErrMsgIdAndTxt(errId, "Wrong parameter count (4 expected).");

	mxArrayAdapter volume(prhs[0]);
	int3 dimsvolume = MWDimsToInt3(mxGetNumberOfDimensions(volume.underlyingarray), mxGetDimensions(volume.underlyingarray));
	if (dimsvolume.x != dimsvolume.y || dimsvolume.x != dimsvolume.z)
		mexErrMsgIdAndTxt(errId, "Volume must be a cube.");
	tfloat* d_volume = volume.GetAsManagedDeviceTFloat();
	d_RemapFull2FullFFT(d_volume, d_volume, dimsvolume);

	mxArrayAdapter volumepsf(prhs[1]);
	tfloat* d_volumepsf = volumepsf.GetAsManagedDeviceTFloat();

	mxArrayAdapter angles(prhs[2]);
	int3 dimsangles = MWDimsToInt3(mxGetNumberOfDimensions(angles.underlyingarray), mxGetDimensions(angles.underlyingarray));
	if (dimsangles.x != 3)
		mexErrMsgIdAndTxt(errId, "3 values per column expected for angles.");
	int batch = dimsangles.y;
	tfloat3* h_angles = (tfloat3*)angles.GetAsManagedTFloat();

	mxArrayAdapter shifts(prhs[3]);
	int3 dimsshifts = MWDimsToInt3(mxGetNumberOfDimensions(shifts.underlyingarray), mxGetDimensions(shifts.underlyingarray));
	if (dimsshifts.x != 2)
		mexErrMsgIdAndTxt(errId, "2 values per column expected for shifts.");
	if (dimsshifts.y != dimsangles.y)
		mexErrMsgIdAndTxt(errId, "Angles and shifts must have equal Y dimensions.");
	tfloat2* h_shifts = (tfloat2*)shifts.GetAsManagedTFloat();

	int3 dimsproj = toInt3(dimsvolume.x, dimsvolume.x, 1);
	tfloat* d_proj;
	cudaMalloc((void**)&d_proj, Elements(dimsproj) * batch * sizeof(tfloat));
	tfloat* d_projpsf;
	cudaMalloc((void**)&d_projpsf, ElementsFFT(dimsproj) * batch * sizeof(tfloat));

	d_ProjForward(d_volume, d_volumepsf, dimsvolume, d_proj, d_projpsf, h_angles, h_shifts, T_INTERP_SINC, batch);

	d_RemapFullFFT2Full(d_proj, d_proj, dimsproj, batch);
	d_RemapHalfFFT2Half(d_projpsf, d_projpsf, dimsproj, batch);

	mwSize outputdims[3];
	outputdims[0] = dimsproj.x;
	outputdims[1] = dimsproj.y;
	outputdims[2] = batch;
	mxArrayAdapter A(mxCreateNumericArray(3,
					outputdims,
					mxGetClassID(volume.underlyingarray),
					mxREAL));
	A.SetFromDeviceTFloat(d_proj);
	cudaFree(d_proj);
	plhs[0] = A.underlyingarray;

	outputdims[0] = dimsproj.x / 2 + 1;
	outputdims[1] = dimsproj.y;
	outputdims[2] = batch;
	mxArrayAdapter B(mxCreateNumericArray(3,
		outputdims,
		mxGetClassID(volume.underlyingarray),
		mxREAL));
	B.SetFromDeviceTFloat(d_projpsf);
	cudaFree(d_projpsf);
	plhs[1] = B.underlyingarray;
}