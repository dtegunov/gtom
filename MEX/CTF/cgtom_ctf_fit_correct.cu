#include "..\Prerequisites.h"


void mexFunction(int nlhs, mxArray *plhs[],
	int nrhs, mxArray const *prhs[])
{
	char const * const errId = "GTOM:CTF:FitCorrect:InvalidInput";

	mxInitGPU();

	if (nrhs != 4)
		mexErrMsgIdAndTxt(errId, "Wrong parameter count (4 expected).");

	mxArrayAdapter image(prhs[0]);
	int3 dimsimage = MWDimsToInt3(mxGetNumberOfDimensions(image.underlyingarray), mxGetDimensions(image.underlyingarray));
	tfloat* d_image = image.GetAsManagedDeviceTFloat();

	mxArrayAdapter fparams(prhs[1]);
	int3 dimsfparams = MWDimsToInt3(mxGetNumberOfDimensions(fparams.underlyingarray), mxGetDimensions(fparams.underlyingarray));
	if (dimsfparams.x != 3 || dimsfparams.y != 11)
		mexErrMsgIdAndTxt(errId, "Fitting parameters should be a 3x11 matrix.");
	tfloat3* h_fparams = (tfloat3*)fparams.GetAsManagedTFloat();
	CTFFitParams fp;
	fp.pixelsize = h_fparams[0];
	fp.Cs = h_fparams[1];
	fp.Cc = h_fparams[2];
	fp.voltage = h_fparams[3];
	fp.defocus = h_fparams[4];
	fp.astigmatismangle = h_fparams[5];
	fp.defocusdelta = h_fparams[6];
	fp.amplitude = h_fparams[7];
	fp.Bfactor = h_fparams[8];
	fp.decayCohIll = h_fparams[9];
	fp.decayspread = h_fparams[10];

	mxArrayAdapter fparamsint(prhs[2]);
	int3 dimsfparamsint = MWDimsToInt3(mxGetNumberOfDimensions(fparamsint.underlyingarray), mxGetDimensions(fparamsint.underlyingarray));
	if (dimsfparamsint.x != 3)
		mexErrMsgIdAndTxt(errId, "Kernel size parameters should contain 3 elements (kernel size, inner radius, outer radius).");
	tfloat* h_fparamsint = fparamsint.GetAsManagedTFloat();
	fp.dimsperiodogram = toInt2((int)(h_fparamsint[0] + 0.5f), (int)(h_fparamsint[0] + 0.5f));
	fp.maskinnerradius = (int)(h_fparamsint[1] + 0.5f);
	fp.maskouterradius = (int)(h_fparamsint[2] + 0.5f);

	mxArrayAdapter params(prhs[3]);
	int3 dimsparams = MWDimsToInt3(mxGetNumberOfDimensions(params.underlyingarray), mxGetDimensions(params.underlyingarray));
	if (dimsparams.x != 11)
		mexErrMsgIdAndTxt(errId, "Start parameters should contain 11 elements.");
	tfloat* h_params = params.GetAsManagedTFloat();
	CTFParams p;
	p.pixelsize = h_params[0];
	p.Cs = h_params[1];
	p.Cc = h_params[2];
	p.voltage = h_params[3];
	p.defocus = h_params[4];
	p.astigmatismangle = h_params[5];
	p.defocusdelta = h_params[6];
	p.amplitude = h_params[7];
	p.Bfactor = h_params[8];
	p.decayCohIll = h_params[9];
	p.decayspread = h_params[10];

	// Fit
	CTFParams fitresult;
	tfloat score, mean, stddev;
	d_CTFFit(d_image, toInt2(dimsimage), 0.5f, p, fp, 3, fitresult, score, mean, stddev);

	// Adjust initial parameters
	for (uint i = 0; i < 11; i++)
		((tfloat*)&p)[i] += ((tfloat*)&fitresult)[i];

	// Correct using current fit
	tcomplex* d_imageft;
	cudaMalloc((void**)&d_imageft, ElementsFFT(dimsimage) * sizeof(tcomplex));
	d_FFTR2C(d_image, d_imageft, 2, dimsimage);

	d_CTFCorrect(d_imageft, dimsimage, p, d_imageft);

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
	B.SetFromTFloat((tfloat*)&p);
	plhs[1] = B.underlyingarray;
}