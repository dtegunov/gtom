#include "..\Prerequisites.h"


void mexFunction(int nlhs, mxArray *plhs[],
	int nrhs, mxArray const *prhs[])
{
	char const * const errId = "GTOM:Projection:ForwardProj:InvalidInput";

	mxInitGPU();

	if (nrhs < 2)
		mexErrMsgIdAndTxt(errId, "Not enough parameters (2 or 3 expected).");

	mxArrayAdapter volume(prhs[0]);
	int ndims = mxGetNumberOfDimensions(volume.underlyingarray);
	int3 dimsvolume = MWDimsToInt3(ndims, mxGetDimensions(volume.underlyingarray));
	if (dimsvolume.x != dimsvolume.y || dimsvolume.x != dimsvolume.z)
		mexErrMsgIdAndTxt(errId, "Volume must be a cube.");
	tfloat* d_volume = volume.GetAsManagedDeviceTFloat();

	mxArrayAdapter angles(prhs[1]);
	ndims = mxGetNumberOfDimensions(angles.underlyingarray);
	int3 dimsangles = MWDimsToInt3(ndims, mxGetDimensions(angles.underlyingarray));
	if (dimsangles.x != 3)
		mexErrMsgIdAndTxt(errId, "3 values per column expected for angles.");
	tfloat3* h_angles = (tfloat3*)angles.GetAsManagedTFloat();

	short kernelsize = 10;
	if (nrhs == 3)
	{
		mxArrayAdapter kernel(prhs[2]);
		tfloat* h_kernel = kernel.GetAsManagedTFloat();
		kernelsize = (short)(h_kernel[0] + 0.5f);
	}
	if (kernelsize < 0 || kernelsize > dimsvolume.x / 2)
		mexErrMsgIdAndTxt(errId, "Kernel size should be between 0 and half of the volume side length.");

	int3 dimsproj = toInt3(dimsvolume.x, dimsvolume.x, 1);
	tfloat* d_proj;
	cudaMalloc((void**)&d_proj, Elements(dimsproj) * dimsangles.y * sizeof(tfloat));

	d_RemapFullFFT2Full(d_volume, d_volume, dimsvolume);
	d_ProjForward(d_volume, dimsvolume, d_proj, dimsproj, h_angles, kernelsize, dimsangles.y);
	d_RemapFull2FullFFT(d_proj, d_proj, dimsproj, dimsangles.y);

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