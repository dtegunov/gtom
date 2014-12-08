#include "..\Prerequisites.h"


void mexFunction(int nlhs, mxArray *plhs[],
	int nrhs, mxArray const *prhs[])
{
	char const * const errId = "GTOM:CTF:FitCorrect:InvalidInput";

	mxInitGPU();

	if (nrhs != 3)
		mexErrMsgIdAndTxt(errId, "Wrong parameter count (3 expected).");

	mxArrayAdapter image(prhs[0]);
	int3 dimsimage = MWDimsToInt3(mxGetNumberOfDimensions(image.underlyingarray), mxGetDimensions(image.underlyingarray));
	tfloat* d_image = image.GetAsManagedDeviceTFloat();

	mxArrayAdapter params(prhs[1]);
	int3 dimsparams = MWDimsToInt3(mxGetNumberOfDimensions(params.underlyingarray), mxGetDimensions(params.underlyingarray));
	if (dimsparams.x != 3 || dimsparams.y != 11)
		mexErrMsgIdAndTxt(errId, "Fitting parameters should be a 3x11 matrix.");
	tfloat3* h_params = (tfloat3*)params.GetAsManagedTFloat();
	CTFFitParams fp;
	fp.pixelsize = h_params[0];
	fp.Cs = h_params[1];
	fp.Cc = h_params[2];
	fp.voltage = h_params[3];
	fp.defocus = h_params[4];
	fp.astigmatismangle = h_params[5];
	fp.defocusdelta = h_params[6];
	fp.amplitude = h_params[7];
	fp.Bfactor = h_params[8];
	fp.decayCohIll = h_params[9];
	fp.decayspread = h_params[10];

	mxArrayAdapter paramsint(prhs[2]);
	int3 dimsparamsint = MWDimsToInt3(mxGetNumberOfDimensions(paramsint.underlyingarray), mxGetDimensions(paramsint.underlyingarray));
	if (dimsparams.x != 3)
		mexErrMsgIdAndTxt(errId, "Kernel size parameters should contain 3 elements.");
	tfloat* h_paramsint = paramsint.GetAsManagedTFloat();
	fp.dimsperiodogram = toInt2((int)(h_paramsint[0] + 0.5f), (int)(h_paramsint[0] + 0.5f));
	fp.maskinnerradius = (int)(h_paramsint[1] + 0.5f);
	fp.maskouterradius = (int)(h_paramsint[2] + 0.5f);

	CTFParams fitresult;
	tfloat score, mean, stddev;
	d_CTFFit(d_image, toInt2(dimsimage), NULL, 0, fp, 2, fitresult, score, mean, stddev);

	tcomplex* d_imageft;
	cudaMalloc((void**)&d_imageft, ElementsFFT(dimsimage) * sizeof(tcomplex));
	d_FFTR2C(d_image, d_imageft, 2, dimsimage);

	d_CTFCorrect(d_imageft, dimsimage, fitresult, d_imageft);

	d_IFFTC2R(d_imageft, d_image, 2, dimsimage);
	cudaFree(d_imageft);

	mwSize outputdims[3];
	outputdims[0] = dimsimage.x;
	outputdims[1] = dimsimage.y;
	outputdims[2] = 1;
	mxArrayAdapter A(mxCreateNumericArray(3,
		outputdims,
		mxGetClassID(image.underlyingarray),
		mxREAL));
	A.SetFromDeviceTFloat(d_image);
	plhs[0] = A.underlyingarray;

	mwSize fitdims[3];
	fitdims[0] = 11;
	fitdims[1] = 1;
	fitdims[2] = 1;
	mxArrayAdapter B(mxCreateNumericArray(1,
		fitdims,
		mxSINGLE_CLASS,
		mxREAL));
	B.SetFromTFloat((tfloat*)&fitresult);
	plhs[1] = B.underlyingarray;
}