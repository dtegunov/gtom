#include "..\Prerequisites.h"

//0: Volume dimensions
//1: Volume offset
//2: Images
//3: Image angles
//4: Image offsets
//5: Image scales

void mexFunction(int nlhs, mxArray *plhs[],
	int nrhs, mxArray const *prhs[])
{
	char const * const errId = "GTOM:Projection:BackwardProj:InvalidInput";

	mxInitGPU();

	if (nrhs < 6)
		mexErrMsgIdAndTxt(errId, "Not enough parameters (6 expected).");

	mxArrayAdapter volumesize(prhs[0]);
	tfloat* h_volumesize = volumesize.GetAsManagedTFloat();
	int3 dimsvolume = toInt3((int)(h_volumesize[0] + 0.5f), (int)(h_volumesize[1] + 0.5f), (int)(h_volumesize[2] + 0.5f));

	tfloat* d_volume;
	cudaMalloc((void**)&d_volume, Elements(dimsvolume) * sizeof(tfloat));

	mxArrayAdapter volumeoffset(prhs[1]);
	tfloat* h_volumeoffset = volumeoffset.GetAsManagedTFloat();
	tfloat3 offset(h_volumeoffset[0], h_volumeoffset[1], h_volumeoffset[2]);

	mxArrayAdapter images(prhs[2]);
	int ndims = mxGetNumberOfDimensions(images.underlyingarray);
	int3 dimsimage = MWDimsToInt3(ndims, mxGetDimensions(images.underlyingarray));
	int nimages = dimsimage.z;
	dimsimage.z = 1;
	tfloat* d_images = images.GetAsManagedDeviceTFloat();

	mxArrayAdapter angles(prhs[3]);
	ndims = mxGetNumberOfDimensions(angles.underlyingarray);
	int3 dimsangles = MWDimsToInt3(ndims, mxGetDimensions(angles.underlyingarray));
	if (dimsangles.x != 3)
		mexErrMsgIdAndTxt(errId, "3 values per column expected for angles.");
	tfloat3* h_angles = (tfloat3*)angles.GetAsManagedTFloat();

	mxArrayAdapter offsets(prhs[4]);
	ndims = mxGetNumberOfDimensions(offsets.underlyingarray);
	int3 dimsoffsets = MWDimsToInt3(ndims, mxGetDimensions(offsets.underlyingarray));
	if (dimsoffsets.x != 2)
		mexErrMsgIdAndTxt(errId, "2 values per column expected for offsets.");
	tfloat2* h_offsets = (tfloat2*)offsets.GetAsManagedTFloat();

	mxArrayAdapter scales(prhs[5]);
	ndims = mxGetNumberOfDimensions(scales.underlyingarray);
	int3 dimsscales = MWDimsToInt3(ndims, mxGetDimensions(scales.underlyingarray));
	if (dimsscales.x != 2)
		mexErrMsgIdAndTxt(errId, "2 values per column expected for scales.");
	tfloat2* h_scales = (tfloat2*)scales.GetAsManagedTFloat();

	d_ProjBackward(d_volume, dimsvolume, offset, d_images, dimsimage, h_angles, h_offsets, h_scales, T_INTERP_SINC, true, nimages);

	mwSize outputdims[3];
	outputdims[0] = dimsvolume.x;
	outputdims[1] = dimsvolume.y;
	outputdims[2] = dimsvolume.z;
	mxArrayAdapter A(mxCreateNumericArray(3,
		outputdims,
		mxSINGLE_CLASS,
		mxREAL));
	A.SetFromDeviceTFloat(d_volume);
	cudaFree(d_volume);
	plhs[0] = A.underlyingarray;
}