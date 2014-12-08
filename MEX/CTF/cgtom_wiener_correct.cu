#include "..\Prerequisites.h"


void mexFunction(int nlhs, mxArray *plhs[],
	int nrhs, mxArray const *prhs[])
{
	char const * const errId = "GTOM:CTF:WienerCorrect:InvalidInput";

	mxInitGPU();

	if (nrhs != 3)
		mexErrMsgIdAndTxt(errId, "Wrong parameter count (3 expected).");

	mxArrayAdapter image(prhs[0]);
	int3 dimsimage = MWDimsToInt3(mxGetNumberOfDimensions(image.underlyingarray), mxGetDimensions(image.underlyingarray));
	tfloat* d_image = image.GetAsManagedDeviceTFloat();
	int nimages = dimsimage.z;
	dimsimage.z = 1;

	mxArrayAdapter params(prhs[1]);
	int3 dimsparams = MWDimsToInt3(mxGetNumberOfDimensions(params.underlyingarray), mxGetDimensions(params.underlyingarray));
	if (dimsparams.x % 11 != 0)
		mexErrMsgIdAndTxt(errId, "CTF parameters should have 11 elements per image.");
	CTFParams* h_params = (CTFParams*)params.GetAsManagedTFloat();

	mxArrayAdapter fsc(prhs[2]);
	int3 dimsfsc = MWDimsToInt3(mxGetNumberOfDimensions(fsc.underlyingarray), mxGetDimensions(fsc.underlyingarray));
	if (dimsfsc.x != dimsimage.x / 2 || dimsfsc.y != nimages)
		mexErrMsgIdAndTxt(errId, "FSC curve matrix should have [size(image, 1) / 2] elements in first, and [size(image, 3)] elements in second dimension.");
	tfloat* d_fsc = fsc.GetAsManagedDeviceTFloat();

	tcomplex* d_imageft;
	cudaMalloc((void**)&d_imageft, ElementsFFT(dimsimage) * nimages * sizeof(tcomplex));
	tfloat* d_weights;
	cudaMalloc((void**)&d_weights, ElementsFFT(dimsimage) * nimages * sizeof(tfloat));

	d_FFTR2C(d_image, d_imageft, 2, dimsimage, nimages);

	for (int n = 0; n < nimages; n++)
		d_WienerPerFreq(d_imageft + ElementsFFT(dimsimage) * n, dimsimage, d_fsc + (dimsimage.x / 2) * n, h_params[n], d_imageft + ElementsFFT(dimsimage) * n, d_weights + ElementsFFT(dimsimage) * n);

	d_IFFTC2R(d_imageft, d_image, 2, dimsimage, nimages);

	mwSize outputdims[3];
	outputdims[0] = dimsimage.x;
	outputdims[1] = dimsimage.y;
	outputdims[2] = nimages;
	mxArrayAdapter A(mxCreateNumericArray(3,
		outputdims,
		mxGetClassID(image.underlyingarray),
		mxREAL));
	A.SetFromDeviceTFloat(d_image);
	plhs[0] = A.underlyingarray;

	mwSize fitdims[3];
	fitdims[0] = dimsimage.x / 2 + 1;
	fitdims[1] = dimsimage.y;
	fitdims[2] = nimages;
	mxArrayAdapter B(mxCreateNumericArray(3,
		fitdims,
		mxSINGLE_CLASS,
		mxREAL));
	B.SetFromDeviceTFloat(d_weights);
	plhs[1] = B.underlyingarray;

	cudaFree(d_imageft);
	cudaFree(d_weights);
}