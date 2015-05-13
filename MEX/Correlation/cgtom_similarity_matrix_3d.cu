#include "..\Prerequisites.h"


void mexFunction(int nlhs, mxArray *plhs[],
	int nrhs, mxArray const *prhs[])
{
	char const * const errId = "GTOM:Correlation:SimilarityMatrix3D:InvalidInput";

	mxInitGPU();

	if (nrhs != 4)
		mexErrMsgIdAndTxt(errId, "Wrong parameter count (4 expected).");

	mxArrayAdapter volumes(prhs[0]);
	int3 dimsvolumes = MWDimsToInt3(mxGetNumberOfDimensions(volumes.underlyingarray), mxGetDimensions(volumes.underlyingarray));
	int nvolumes = dimsvolumes.z / dimsvolumes.x;
	dimsvolumes.z = dimsvolumes.x;
	tfloat* d_volumes = volumes.GetAsManagedDeviceTFloat();

	mxArrayAdapter psf(prhs[1]);
	tfloat* d_psf = psf.GetAsManagedDeviceTFloat();

	d_NormMonolithic(d_volumes, d_volumes, Elements(dimsvolumes), T_NORM_MEAN01STD, nvolumes);
	d_HannMask(d_volumes, d_volumes, dimsvolumes, NULL, NULL, nvolumes);
	d_RemapFull2FullFFT(d_volumes, d_volumes, dimsvolumes, nvolumes);

	mxArrayAdapter a_angularspacing(prhs[2]);
	tfloat* h_angularspacing = a_angularspacing.GetAsManagedTFloat();

	mxArrayAdapter a_maxtheta(prhs[3]);
	tfloat* h_maxtheta = a_maxtheta.GetAsManagedTFloat();

	tfloat* h_simmatrix = MallocValueFilled(nvolumes * nvolumes, (tfloat)0);
	tfloat* h_rotmatrix = MallocValueFilled(nvolumes * nvolumes * 3, (tfloat)0);
	tfloat* h_transmatrix = MallocValueFilled(nvolumes * nvolumes * 3, (tfloat)0);
	tfloat* h_samplesmatrix = MallocValueFilled(nvolumes * nvolumes, (tfloat)0);
	Align3DParams* h_results = (Align3DParams*)malloc(nvolumes * sizeof(Align3DParams));

	for (int i = 0; i < nvolumes - 1; i++)
	{
		int elementsoffset = i + 1;
		int rowelements = nvolumes - elementsoffset;

		d_Align3D(d_volumes + Elements(dimsvolumes) * elementsoffset, d_psf + ElementsFFT(dimsvolumes) * elementsoffset,
			d_volumes + Elements(dimsvolumes) * i, d_psf + ElementsFFT(dimsvolumes) * i,
			NULL,
			dimsvolumes,
			rowelements,
			h_angularspacing[0],
			tfloat2(0, PI2), tfloat2(-h_maxtheta[0], h_maxtheta[0]), tfloat2(0, PI2),
			true,
			h_results + elementsoffset);

		for (int j = elementsoffset; j < nvolumes; j++)
		{
			h_simmatrix[j * nvolumes + i] = h_results[j].score;
			h_simmatrix[i * nvolumes + j] = h_results[j].score;

			h_samplesmatrix[j * nvolumes + i] = h_results[j].samples;
			h_samplesmatrix[i * nvolumes + j] = h_results[j].samples;


			tfloat3 trans = h_results[j].translation;
			glm::vec3 v_trans = glm::inverse(Matrix3Euler(h_results[j].rotation)) * glm::vec3(trans.x, trans.y, trans.z);
			h_transmatrix[i * nvolumes + j] = v_trans.x;
			h_transmatrix[nvolumes * nvolumes + i * nvolumes + j] = v_trans.y;
			h_transmatrix[nvolumes * nvolumes * 2 + i * nvolumes + j] = v_trans.z;

			h_transmatrix[j * nvolumes + i] = -v_trans.x;
			h_transmatrix[nvolumes * nvolumes + j * nvolumes + i] = -v_trans.y;
			h_transmatrix[nvolumes * nvolumes * 2 + j * nvolumes + i] = -v_trans.z;


			tfloat3 rot = h_results[j].rotation;
			h_rotmatrix[i * nvolumes + j] = rot.x;
			h_rotmatrix[nvolumes * nvolumes + i * nvolumes + j] = rot.y;
			h_rotmatrix[nvolumes * nvolumes * 2 + i * nvolumes + j] = rot.z;

			tfloat3 invrot = EulerInverse(rot);
			h_rotmatrix[j * nvolumes + i] = invrot.x;
			h_rotmatrix[nvolumes * nvolumes + j * nvolumes + i] = invrot.y;
			h_rotmatrix[nvolumes * nvolumes * 2 + j * nvolumes + i] = invrot.z;
		}

		mexPrintf("%d\n", i);
		mexEvalString("drawnow;");
	}

	mwSize outputdims[3];
	outputdims[0] = nvolumes;
	outputdims[1] = nvolumes;
	outputdims[2] = 1;
	mxArrayAdapter A(mxCreateNumericArray(2,
		outputdims,
		mxGetClassID(volumes.underlyingarray),
		mxREAL));
	A.SetFromTFloat(h_simmatrix);
	plhs[0] = A.underlyingarray;

	outputdims[0] = nvolumes;
	outputdims[1] = nvolumes;
	outputdims[2] = 1;
	mxArrayAdapter B(mxCreateNumericArray(2,
		outputdims,
		mxGetClassID(volumes.underlyingarray),
		mxREAL));
	B.SetFromTFloat(h_samplesmatrix);
	plhs[1] = B.underlyingarray;

	outputdims[0] = nvolumes;
	outputdims[1] = nvolumes;
	outputdims[2] = 3;
	mxArrayAdapter C(mxCreateNumericArray(3,
		outputdims,
		mxGetClassID(volumes.underlyingarray),
		mxREAL));
	C.SetFromTFloat(h_rotmatrix);
	plhs[2] = C.underlyingarray;

	outputdims[0] = nvolumes;
	outputdims[1] = nvolumes;
	outputdims[2] = 3;
	mxArrayAdapter D(mxCreateNumericArray(3,
		outputdims,
		mxGetClassID(volumes.underlyingarray),
		mxREAL));
	D.SetFromTFloat(h_transmatrix);
	plhs[3] = D.underlyingarray;

	free(h_samplesmatrix);
	free(h_transmatrix);
	free(h_rotmatrix);
	free(h_simmatrix);
}