#include "..\Prerequisites.h"
using namespace gtom;

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
	if (dimsfparams.x != 3 || dimsfparams.y != 10)
		mexErrMsgIdAndTxt(errId, "Fitting parameters should be a 3x10 matrix.");
	tfloat3* h_fparams = (tfloat3*)fparams.GetAsManagedTFloat();
	CTFFitParams fp;
	fp.pixelsize = h_fparams[0];
	fp.Cs = h_fparams[1];
	fp.voltage = h_fparams[2];
	fp.defocus = h_fparams[3];
	fp.astigmatismangle = h_fparams[4];
	fp.defocusdelta = h_fparams[5];
	fp.amplitude = h_fparams[6];
	fp.Bfactor = h_fparams[7];
	fp.scale = h_fparams[8];
	fp.phaseshift = h_fparams[9];

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
	if (dimsparams.x != 10)
		mexErrMsgIdAndTxt(errId, "Start parameters should contain 10 elements.");
	tfloat* h_params = params.GetAsManagedTFloat();
	CTFParams p;
	p.pixelsize = h_params[0];
	p.Cs = h_params[1];
	p.voltage = h_params[2];
	p.defocus = h_params[3];
	p.astigmatismangle = h_params[4];
	p.defocusdelta = h_params[5];
	p.amplitude = h_params[6];
	p.Bfactor = h_params[7];
	p.scale = h_params[8];
	p.phaseshift = h_params[9];

	// Fit
	CTFParams fitresult = p;
	tfloat score, mean, stddev;
	d_CTFFit(d_image, toInt2(dimsimage), 0.5f, p, fp, 3, fitresult, score, mean, stddev);

	// Adjust initial parameters
	for (uint i = 0; i < 10; i++)
		((tfloat*)&p)[i] += ((tfloat*)&fitresult)[i];

	mwSize fitdims[3];
	fitdims[0] = 10;
	fitdims[1] = 1;
	fitdims[2] = 1;
	mxArrayAdapter A(mxCreateNumericArray(1,
		fitdims,
		mxSINGLE_CLASS,
		mxREAL));
	A.SetFromTFloat((tfloat*)&p);
	plhs[0] = A.underlyingarray;
}