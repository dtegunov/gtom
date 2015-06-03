#include "..\Prerequisites.h"
using namespace gtom;

void mexFunction(int nlhs, mxArray *plhs[],
	int nrhs, mxArray const *prhs[])
{
	char const * const errId = "GTOM:CTF:TiltCorrect:InvalidInput";

	mxInitGPU();

	if (nrhs != 5)
		mexErrMsgIdAndTxt(errId, "Wrong parameter count (5 expected).");

	mxArrayAdapter image(prhs[0]);
	int3 dimsimage = MWDimsToInt3(mxGetNumberOfDimensions(image.underlyingarray), mxGetDimensions(image.underlyingarray));
	uint nimages = dimsimage.z;
	dimsimage.z = 1;
	tfloat* h_image = image.GetAsManagedTFloat();

	vector<CTFTiltParams> v_tiltparams;

	mxArrayAdapter specimentilt(prhs[1]);

	mxArrayAdapter stagetilts(prhs[2]);
	int3 dimstilts = MWDimsToInt3(mxGetNumberOfDimensions(stagetilts.underlyingarray), mxGetDimensions(stagetilts.underlyingarray));
	if (dimstilts.x != 2 || dimstilts.y != nimages)
		mexErrMsgIdAndTxt(errId, "Stage tilts should be a matrix with 2x[nimages] elements.");
	tfloat2* h_stagetilts = (tfloat2*)stagetilts.GetAsManagedTFloat();
	for (uint n = 0; n < nimages; n++)
	{
		CTFTiltParams tiltparams(0.0f, h_stagetilts[n], ((tfloat2*)specimentilt.GetAsManagedTFloat())[0], CTFParams());
		v_tiltparams.push_back(tiltparams);
	}

	mxArrayAdapter params(prhs[3]);
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
		v_tiltparams[n].centerparams = p;
	}

	mxArrayAdapter snr(prhs[4]);

	tfloat* h_filtered;
	cudaMallocHost((void**)&h_filtered, Elements2(dimsimage) * nimages * sizeof(tfloat));

	// Correct
	for (uint n = 0; n < nimages; n++)
	{
		tfloat* d_image = (tfloat*)CudaMallocFromHostArray(h_image + Elements2(dimsimage) * n, Elements2(dimsimage) * sizeof(tfloat));
		d_CTFTiltCorrect(d_image, toInt2(dimsimage), v_tiltparams[n], snr.GetAsManagedTFloat()[0], d_image);
		cudaMemcpy(h_filtered + Elements2(dimsimage) * n, d_image, Elements2(dimsimage) * sizeof(tfloat), cudaMemcpyDeviceToHost);
		cudaFree(d_image);
	}

	mwSize outputdims[3];
	outputdims[0] = dimsimage.x;
	outputdims[1] = dimsimage.y;
	outputdims[2] = nimages;
	mxArrayAdapter A(mxCreateNumericArray(3,
		outputdims,
		mxSINGLE_CLASS,
		mxREAL));
	A.SetFromTFloat(h_filtered);
	plhs[0] = A.underlyingarray;
	cudaFreeHost(h_filtered);
}