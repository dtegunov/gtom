#include "..\Prerequisites.h"


void mexFunction(int nlhs, mxArray *plhs[],
	int nrhs, mxArray const *prhs[])
{
	char const * const errId = "GTOM:Reconstruction:RecFourierFinalize:InvalidInput";
	char const * const errMsg = "Invalid input to MEX file.";

	mxInitGPU();

	if (nrhs < 4)
		mexErrMsgIdAndTxt(errId, "Wrong parameter count (4 expected).");

	mxArrayAdapter volumeft(prhs[0]);
	tcomplex* d_volumeft = volumeft.GetAsManagedDeviceTComplex();

	mxArrayAdapter samples(prhs[1]);
	tfloat* d_samples = samples.GetAsManagedDeviceTFloat();

	int3 dimsvolume = toInt3((int)(((double*)mxGetData(prhs[2]))[0] + 0.5), (int)(((double*)mxGetData(prhs[2]))[1] + 0.5), (int)(((double*)mxGetData(prhs[2]))[2] + 0.5));

	mxArrayAdapter lowpass(prhs[3]);
	tfloat* h_lowpass = lowpass.GetAsManagedTFloat();

	d_MaxOp(d_samples, (tfloat)1, d_samples, ElementsFFT(dimsvolume));
	d_Inv(d_samples, d_samples, ElementsFFT(dimsvolume));
	d_ComplexMultiplyByVector(d_volumeft, d_samples, d_volumeft, ElementsFFT(dimsvolume));

	tfloat* d_volume;
	cudaMalloc((void**)&d_volume, Elements(dimsvolume) * sizeof(tfloat));

	d_RemapHalf2HalfFFT(d_volumeft, d_volumeft, dimsvolume);
	d_Bandpass(d_volumeft, d_volumeft, dimsvolume, 0, h_lowpass[0], 0);
	d_IFFTC2R(d_volumeft, d_volume, 3, dimsvolume);
	d_RemapFullFFT2Full(d_volume, d_volume, dimsvolume);

	d_Norm(d_volume, d_volume, Elements(dimsvolume), (tfloat*)NULL, T_NORM_MEAN01STD, 0);

	mwSize output1dims[3];
	output1dims[0] = dimsvolume.x;
	output1dims[1] = dimsvolume.y;
	output1dims[2] = dimsvolume.z;
	mxArrayAdapter A(mxCreateNumericArray(3,
		output1dims,
		mxGetClassID(samples.underlyingarray),
		mxREAL));
	A.SetFromDeviceTFloat(d_volume);
	plhs[0] = A.underlyingarray;

	cudaFree(d_volume);
}