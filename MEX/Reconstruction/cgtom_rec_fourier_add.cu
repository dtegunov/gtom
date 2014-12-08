#include "..\Prerequisites.h"


void mexFunction(int nlhs, mxArray *plhs[],
	int nrhs, mxArray const *prhs[])
{
	char const * const errId = "GTOM:Reconstruction:RecFourierAdd:InvalidInput";
	char const * const errMsg = "Invalid input to MEX file.";

	mxInitGPU();

	if (nrhs < 7)
		mexErrMsgIdAndTxt(errId, "Wrong parameter count (6 expected).");

	mxArrayAdapter volumeft(prhs[0]);
	tcomplex* d_volumeft = volumeft.GetAsManagedDeviceTComplex();

	mxArrayAdapter samples(prhs[1]);
	tfloat* d_samples = samples.GetAsManagedDeviceTFloat();

	mxArrayAdapter proj(prhs[2]);
	int ndims = mxGetNumberOfDimensions(proj.underlyingarray);
	int3 dimsproj = MWDimsToInt3(ndims, mxGetDimensions(proj.underlyingarray));
	tfloat* d_proj = proj.GetAsManagedDeviceTFloat();

	mxArrayAdapter weights(prhs[3]);
	int3 dimsweights = MWDimsToInt3(mxGetNumberOfDimensions(weights.underlyingarray), mxGetDimensions(weights.underlyingarray));
	tfloat* d_weights = NULL;
	if (ElementsFFT(dimsweights) > 1)
		d_weights = weights.GetAsManagedDeviceTFloat();

	mxArrayAdapter ctf(prhs[4]);
	int3 dimsctf = MWDimsToInt3(mxGetNumberOfDimensions(ctf.underlyingarray), mxGetDimensions(ctf.underlyingarray));
	CTFParams* h_ctf = NULL;
	if (ElementsFFT(dimsctf) > 1)
		h_ctf = (CTFParams*)ctf.GetAsManagedTFloat();

	mxArrayAdapter angles(prhs[5]);
	ndims = mxGetNumberOfDimensions(angles.underlyingarray);
	int3 dimsangles = MWDimsToInt3(ndims, mxGetDimensions(angles.underlyingarray));
	if (dimsangles.x != 3)
		mexErrMsgIdAndTxt(errId, "Angles must be a 3 x n sized matrix.");
	tfloat3* h_angles = (tfloat3*)angles.GetAsManagedTFloat();

	int3 dimsvolume = toInt3((int)(((double*)mxGetData(prhs[6]))[0] + 0.5), (int)(((double*)mxGetData(prhs[6]))[1] + 0.5), (int)(((double*)mxGetData(prhs[6]))[2] + 0.5));

	d_ReconstructFourierAdd(d_volumeft, d_samples, d_proj, d_weights, h_ctf, dimsproj, dimsvolume, h_angles);

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