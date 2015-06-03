#include "..\Prerequisites.h"
using namespace gtom;

void mexFunction(int nlhs, mxArray *plhs[],
	int nrhs, mxArray const *prhs[])
{
	char const * const errId = "GTOM:CTF:TiltFit:InvalidInput";

	mxInitGPU();

	if (nrhs != 6)
		mexErrMsgIdAndTxt(errId, "Wrong parameter count (6 expected).");

	mxArrayAdapter image(prhs[0]);
	int3 dimsimage = MWDimsToInt3(mxGetNumberOfDimensions(image.underlyingarray), mxGetDimensions(image.underlyingarray));
	uint nimages = dimsimage.z;
	dimsimage.z = 1;
	tfloat* h_image = image.GetAsManagedTFloat();

	mxArrayAdapter defocusbracket(prhs[1]);
	int3 dimsdefocusbracket = MWDimsToInt3(mxGetNumberOfDimensions(defocusbracket.underlyingarray), mxGetDimensions(defocusbracket.underlyingarray));
	if (dimsdefocusbracket.x != 3)
		mexErrMsgIdAndTxt(errId, "Defocus bracket should have 3 elements.");
	tfloat3* h_defocusbracket = (tfloat3*)defocusbracket.GetAsManagedTFloat();
	CTFFitParams fp;
	fp.defocus = h_defocusbracket[0];

	mxArrayAdapter fparamsint(prhs[2]);
	int3 dimsfparamsint = MWDimsToInt3(mxGetNumberOfDimensions(fparamsint.underlyingarray), mxGetDimensions(fparamsint.underlyingarray));
	if (dimsfparamsint.x != 3)
		mexErrMsgIdAndTxt(errId, "Kernel size parameters should contain 3 elements (kernel size, inner radius, outer radius).");
	tfloat* h_fparamsint = fparamsint.GetAsManagedTFloat();
	fp.dimsperiodogram = toInt2((int)(h_fparamsint[0] + 0.5f), (int)(h_fparamsint[0] + 0.5f));
	fp.maskinnerradius = (int)(h_fparamsint[1] + 0.5f);
	fp.maskouterradius = (int)(h_fparamsint[2] + 0.5f);

	mxArrayAdapter maxtheta(prhs[3]);

	std::vector<CTFTiltParams> v_startparams;

	mxArrayAdapter stagetilts(prhs[4]);
	int3 dimstilts = MWDimsToInt3(mxGetNumberOfDimensions(stagetilts.underlyingarray), mxGetDimensions(stagetilts.underlyingarray));
	if (dimstilts.x != 2 || dimstilts.y != nimages)
		mexErrMsgIdAndTxt(errId, "Stage tilts should be a matrix with 2x[nimages] elements.");
	tfloat2* h_stagetilts = (tfloat2*)stagetilts.GetAsManagedTFloat();
	for (uint n = 0; n < nimages; n++)
	{
		CTFTiltParams tiltparams(0.0f, h_stagetilts[n], tfloat2(0.0f), CTFParams());
		v_startparams.push_back(tiltparams);
	}

	mxArrayAdapter params(prhs[5]);
	int3 dimsparams = MWDimsToInt3(mxGetNumberOfDimensions(params.underlyingarray), mxGetDimensions(params.underlyingarray));
	if (dimsparams.x != 10 || dimsparams.y != nimages)
		mexErrMsgIdAndTxt(errId, "Start parameters should be a matrix with 10x[nimages] elements.");
	tfloat* h_params = params.GetAsManagedTFloat();
	for (uint n = 0; n < nimages; n++, h_params += 10)
	{
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
		v_startparams[n].centerparams = p;
	}

	// Fit
	tfloat2 specimentilt;
	tfloat* h_defoci = (tfloat*)malloc(nimages * sizeof(tfloat));
	h_CTFTiltFit(h_image, toInt2(dimsimage), nimages, 0.75f, v_startparams, fp, maxtheta.GetAsManagedTFloat()[0], specimentilt, h_defoci);

	// Adjust initial parameters
	for (uint n = 0; n < nimages; n++)
		h_defoci[n] += v_startparams[n].centerparams.defocus;

	mwSize outputdims[3];
	outputdims[0] = nimages;
	outputdims[1] = 1;
	outputdims[2] = 1;
	mxArrayAdapter A(mxCreateNumericArray(1,
		outputdims,
		mxSINGLE_CLASS,
		mxREAL));
	A.SetFromTFloat(h_defoci);
	plhs[0] = A.underlyingarray;
	free(h_defoci);

	mwSize fitdims[3];
	fitdims[0] = 2;
	fitdims[1] = 1;
	fitdims[2] = 1;
	mxArrayAdapter B(mxCreateNumericArray(1,
		fitdims,
		mxSINGLE_CLASS,
		mxREAL));
	B.SetFromTFloat((tfloat*)&specimentilt);
	plhs[1] = B.underlyingarray;
}