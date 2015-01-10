#include "..\Prerequisites.h"


void mexFunction(int nlhs, mxArray *plhs[],
	int nrhs, mxArray const *prhs[])
{
	char const * const errId = "GTOM:Reconstruction:RecFourierAdd:InvalidInput";
	char const * const errMsg = "Invalid input to MEX file.";

	mxInitGPU();

	if (nrhs < 5)
		mexErrMsgIdAndTxt(errId, errMsg);

	mxArrayAdapter volumeft(prhs[0]);
	tcomplex* d_volumeft = volumeft.GetAsManagedDeviceTComplex();

	mxArrayAdapter samples(prhs[1]);
	tfloat* d_samples = samples.GetAsManagedDeviceTFloat();

	mxArrayAdapter proj(prhs[2]);
	int ndims = mxGetNumberOfDimensions(proj.underlyingarray);
	int3 dimsproj = MWDimsToInt3(ndims, mxGetDimensions(proj.underlyingarray));
	tfloat* d_proj = proj.GetAsManagedDeviceTFloat();

	mxArrayAdapter angles(prhs[3]);
	ndims = mxGetNumberOfDimensions(angles.underlyingarray);
	int3 dimsangles = MWDimsToInt3(ndims, mxGetDimensions(angles.underlyingarray));
	tfloat2* h_angles = (tfloat2*)angles.GetAsManagedTFloat();

	int3 dimsvolume = toInt3((int)((double*)mxGetData(prhs[4]))[0], (int)((double*)mxGetData(prhs[4]))[1], (int)((double*)mxGetData(prhs[4]))[2]);

	d_RemapHalfFFT2Half(d_volumeft, d_volumeft, dimsvolume);
	d_RemapHalfFFT2Half(d_samples, d_samples, dimsvolume);
	d_ReconstructFourierAdd(d_volumeft, d_samples, d_proj, dimsproj, dimsvolume, h_angles);
	d_RemapHalf2HalfFFT(d_volumeft, d_volumeft, dimsvolume);
	d_RemapHalf2HalfFFT(d_samples, d_samples, dimsvolume);

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