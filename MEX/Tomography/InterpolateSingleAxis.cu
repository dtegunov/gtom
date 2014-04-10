#include "..\Prerequisites.h"


//(tfloat* interpolated) InterpolateSingleAxis(tfloat* projstack, tfloat* angles, int interpindex, tfloat smoothsigma)
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
    char const * const errId = "GTOM:Tomography:InterpolateSingleAxis:InvalidInput";
    char const * const errMsg = "Invalid input to MEX file.";

    mxInitGPU();

	if (nrhs < 4)
        mexErrMsgIdAndTxt(errId, errMsg);

	mxArrayAdapter proj(prhs[0]);
	int ndims = mxGetNumberOfDimensions(proj.underlyingarray);
	int3 dimsproj = MWDimsToInt3(ndims, mxGetDimensions(proj.underlyingarray));
	tfloat* d_proj = proj.GetAsManagedDeviceTFloat();

	d_RemapFull2FullFFT(d_proj, d_proj, toInt3(dimsproj.x, dimsproj.y, 1), dimsproj.z);

	tcomplex* d_projft;
	cudaMalloc((void**)&d_projft, ElementsFFT(dimsproj) * sizeof(tcomplex));
	d_FFTR2C(d_proj, d_projft, 2, toInt3(dimsproj.x, dimsproj.y, 1), dimsproj.z);
	d_RemapHalfFFT2Half(d_projft, d_projft, toInt3(dimsproj.x, dimsproj.y, 1), dimsproj.z);

	mxArrayAdapter angles(prhs[1]);
	int3 dimsangles = MWDimsToInt3(1, mxGetDimensions(angles.underlyingarray));
	tfloat* h_angles = (tfloat*)angles.GetAsManagedTFloat();

	int interpindex = (int)((double*)mxGetData(prhs[2]))[0];
	tfloat smoothsigma = (tfloat)((double*)mxGetData(prhs[3]))[0];

	tcomplex* d_interpft;
	cudaMalloc((void**)&d_interpft, (dimsproj.x / 2 + 1) * dimsproj.y * sizeof(tcomplex));

	d_InterpolateSingleAxisTilt(d_projft, dimsproj, d_interpft, h_angles, interpindex, smoothsigma);
	cudaFree(d_projft);
	
	d_RemapHalf2HalfFFT(d_interpft, d_interpft, toInt3(dimsproj.x, dimsproj.y, 1));

	tfloat* d_interp;
	cudaMalloc((void**)&d_interp, dimsproj.x * dimsproj.y * sizeof(tfloat));
	d_IFFTC2R(d_interpft, d_interp, 2, toInt3(dimsproj.x, dimsproj.y, 1));
	cudaFree(d_interpft);

	d_RemapFullFFT2Full(d_interp, d_interp, toInt3(dimsproj.x, dimsproj.y, 1));
	
	mwSize outputdims[2];
	outputdims[0] = dimsproj.x;
	outputdims[1] = dimsproj.y;
	mxArrayAdapter B(mxCreateNumericArray(2,
					 outputdims,
					 mxGetClassID(proj.underlyingarray),
					 mxREAL));
	B.SetFromDeviceTFloat(d_interp);
	cudaFree(d_interp);
	plhs[0] = B.underlyingarray;
}