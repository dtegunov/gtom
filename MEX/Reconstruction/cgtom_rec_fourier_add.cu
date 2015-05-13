#include "..\Prerequisites.h"


void mexFunction(int nlhs, mxArray *plhs[],
	int nrhs, mxArray const *prhs[])
{
	char const * const errId = "GTOM:Reconstruction:RecFourierAdd:InvalidInput";
	char const * const errMsg = "Invalid input to MEX file.";

	mxInitGPU();

	if (nrhs < 6)
		mexErrMsgIdAndTxt(errId, "Wrong parameter count (6 expected).");

	mxArrayAdapter volumeft(prhs[0]);
	tcomplex* d_volumeft = volumeft.GetAsManagedDeviceTComplex();

	mxArrayAdapter samples(prhs[1]);
	tfloat* d_samples = samples.GetAsManagedDeviceTFloat();

	mxArrayAdapter proj(prhs[2]);
	int ndims = mxGetNumberOfDimensions(proj.underlyingarray);
	int3 dimsproj = MWDimsToInt3(ndims, mxGetDimensions(proj.underlyingarray));
	int nimages = dimsproj.z;
	dimsproj.z = 1;
	tfloat* d_proj = proj.GetAsManagedDeviceTFloat();
	d_RemapFull2FullFFT(d_proj, d_proj, dimsproj, nimages);

	mxArrayAdapter psf(prhs[3]);
	tfloat* d_psf = psf.GetAsManagedDeviceTFloat();

	mxArrayAdapter angles(prhs[4]);
	int3 dimsangles = MWDimsToInt3(mxGetNumberOfDimensions(angles.underlyingarray), mxGetDimensions(angles.underlyingarray));
	if (dimsangles.x != 3)
		mexErrMsgIdAndTxt(errId, "Angles must be a 3 x n sized matrix.");
	tfloat3* h_angles = (tfloat3*)angles.GetAsManagedTFloat();

	mxArrayAdapter shifts(prhs[5]);
	int3 dimsshifts = MWDimsToInt3(mxGetNumberOfDimensions(shifts.underlyingarray), mxGetDimensions(shifts.underlyingarray));
	if (dimsshifts.x != 3)
		mexErrMsgIdAndTxt(errId, "Shifts must be a 3 x n sized matrix.");
	tfloat2* h_shifts = (tfloat2*)shifts.GetAsManagedTFloat();

	int3 dimsvolume = toInt3(dimsproj.x, dimsproj.x, dimsproj.x);

	d_ReconstructFourierPreciseAdd(d_volumeft, d_samples, dimsproj, d_proj, d_psf, T_INTERP_CUBIC, h_angles, h_shifts, nimages);

	mwSize output1dims[3];
	output1dims[0] = dimsvolume.x / 2 + 1;
	output1dims[1] = dimsvolume.y;
	output1dims[2] = dimsvolume.z;
	mxArrayAdapter A(mxCreateNumericArray(3,
					output1dims,
					mxGetClassID(volumeft.underlyingarray),
					mxCOMPLEX));
	A.SetFromDeviceTComplex(d_volumeft);
	plhs[0] = A.underlyingarray;

	mwSize output2dims[3];
	output2dims[0] = dimsvolume.x / 2 + 1;
	output2dims[1] = dimsvolume.y;
	output2dims[2] = dimsvolume.z;
	mxArrayAdapter B(mxCreateNumericArray(3,
					output2dims,
					mxGetClassID(samples.underlyingarray),
					mxREAL));
	B.SetFromDeviceTFloat(d_samples);
	plhs[1] = B.underlyingarray;
}