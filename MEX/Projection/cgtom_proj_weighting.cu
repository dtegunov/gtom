#include "..\Prerequisites.h"


void mexFunction(int nlhs, mxArray *plhs[],
	int nrhs, mxArray const *prhs[])
{
	char const * const errId = "GTOM:Projection:Weighting:InvalidInput";

	mxInitGPU();

	if (nrhs < 2)
		mexErrMsgIdAndTxt(errId, "Not enough parameters (2 expected).");

	mxArrayAdapter imagesize(prhs[0]);
	int3 dimsimagesize = MWDimsToInt3(mxGetNumberOfDimensions(imagesize.underlyingarray), mxGetDimensions(imagesize.underlyingarray));
	tfloat* h_imagesize = imagesize.GetAsManagedTFloat();
	int3 dimsimage;
	if (max(dimsimagesize.x, dimsimagesize.y) == 2)
		dimsimage = toInt3((int)(h_imagesize[0] + 0.5f), (int)(h_imagesize[1] + 0.5f), 1);
	else if(max(dimsimagesize.x, dimsimagesize.y) == 3)
		dimsimage = toInt3((int)(h_imagesize[0] + 0.5f), (int)(h_imagesize[1] + 0.5f), (int)(h_imagesize[2] + 0.5f));

	mxArrayAdapter angles(prhs[1]);
	int ndims = mxGetNumberOfDimensions(angles.underlyingarray);
	int3 dimsangles = MWDimsToInt3(ndims, mxGetDimensions(angles.underlyingarray));
	if (dimsangles.x != 3)
		mexErrMsgIdAndTxt(errId, "3 values per column expected for angles.");
	tfloat3* h_angles = (tfloat3*)angles.GetAsManagedTFloat();

	int nimages = dimsangles.y;

	mxArrayAdapter indices(prhs[2]);
	tfloat* h_indicesf = (tfloat*)indices.GetAsManagedTFloat();
	int* h_indices = (int*)malloc(nimages * sizeof(int));
	for (int n = 0; n < nimages; n++)
		h_indices[n] = (int)(h_indicesf[n] + 0.5f);

	tfloat* d_weights;
	if (dimsimage.z == 1)
		d_weights = CudaMallocValueFilled((dimsimage.x / 2 + 1) * dimsimage.y * nimages, (tfloat)1);
	else
		d_weights = CudaMallocValueFilled(ElementsFFT(dimsimage), (tfloat)1);

	if (dimsimage.z == 1)
		d_Exact2DWeighting(d_weights, toInt2(dimsimage.x, dimsimage.y), h_indices, h_angles, nimages, dimsimage.x / 2, false);
	else
		d_Exact3DWeighting(d_weights, dimsimage, h_angles, nimages, dimsimage.x / 2, false);

	free(h_indices);

	mwSize outputdims[3];
	outputdims[0] = dimsimage.x / 2 + 1;
	outputdims[1] = dimsimage.y;
	outputdims[2] = dimsimage.z == 1 ? nimages : dimsimage.z;
	mxArrayAdapter A(mxCreateNumericArray(3,
		outputdims,
		mxDOUBLE_CLASS,
		mxREAL));
	A.SetFromDeviceTFloat(d_weights);
	cudaFree(d_weights);
	plhs[0] = A.underlyingarray;
}