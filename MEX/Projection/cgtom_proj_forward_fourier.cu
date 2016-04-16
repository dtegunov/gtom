#include "..\Prerequisites.h"


void mexFunction(int nlhs, mxArray *plhs[],
	int nrhs, mxArray const *prhs[])
{
	char const * const errId = "GTOM:Projection:ForwardProjFourier:InvalidInput";

	mxInitGPU();

	if (nrhs != 2)
		mexErrMsgIdAndTxt(errId, "Wrong parameter count (4 expected).");

	mxArrayAdapter volume(prhs[0]);
	int3 dimsvolume = MWDimsToInt3(mxGetNumberOfDimensions(volume.underlyingarray), mxGetDimensions(volume.underlyingarray));
	if (dimsvolume.x != dimsvolume.y || dimsvolume.x != dimsvolume.z)
		mexErrMsgIdAndTxt(errId, "Volume must be a cube.");
	tfloat* d_volume = volume.GetAsManagedDeviceTFloat();
	d_RemapFull2FullFFT(d_volume, d_volume, dimsvolume);

	tcomplex* d_volumeft;
	cudaMalloc((void**)&d_volumeft, ElementsFFT(dimsvolume) * sizeof(tcomplex));
	d_FFTR2C(d_volume, d_volumeft, 3, dimsvolume, 1);

	mxArrayAdapter angles(prhs[1]);
	int3 dimsangles = MWDimsToInt3(mxGetNumberOfDimensions(angles.underlyingarray), mxGetDimensions(angles.underlyingarray));
	if (dimsangles.x != 3)
		mexErrMsgIdAndTxt(errId, "3 values per column expected for angles.");
	int batch = dimsangles.y;
	tfloat3* h_angles = (tfloat3*)angles.GetAsManagedTFloat();

	int3 dimsproj = toInt3(dimsvolume.x, dimsvolume.x, 1);
	tfloat* d_proj;
	cudaMalloc((void**)&d_proj, Elements(dimsproj) * batch * sizeof(tfloat));
	tcomplex* d_projft;
	cudaMalloc((void**)&d_projft, ElementsFFT(dimsproj) * batch * sizeof(tcomplex));

	d_rlnProject(d_volumeft, dimsvolume, d_projft, dimsproj, h_angles, dimsangles.y);
	cudaFree(d_volumeft);

	d_IFFTC2R(d_projft, d_proj, 2, dimsproj, dimsangles.y);
	d_RemapFullFFT2Full(d_proj, d_proj, dimsproj, batch);

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
	cudaFree(d_projft);
	plhs[0] = A.underlyingarray;

	
}