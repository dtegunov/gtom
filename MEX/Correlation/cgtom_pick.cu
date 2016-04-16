#include "..\Prerequisites.h"


void mexFunction(int nlhs, mxArray *plhs[],
	int nrhs, mxArray const *prhs[])
{
	char const * const errId = "GTOM:Pick:InvalidInput";

	mxInitGPU();

	if (nrhs != 5)
		mexErrMsgIdAndTxt(errId, "Wrong parameter count (5 expected).");

	mxArrayAdapter image(prhs[0]);
	int3 dimsimage = MWDimsToInt3(mxGetNumberOfDimensions(image.underlyingarray), mxGetDimensions(image.underlyingarray));
	tfloat* d_image = image.GetAsManagedDeviceTFloat();

	int ndims = DimensionCount(dimsimage);

	mxArrayAdapter ref(prhs[1]);
	int3 dimsref = MWDimsToInt3(mxGetNumberOfDimensions(ref.underlyingarray), mxGetDimensions(ref.underlyingarray));
	int nrefs = 1;
	if (ndims == 3)
	{
		nrefs = dimsref.z / dimsref.x;
		dimsref.z = dimsref.x;
	}
	else
	{
		nrefs = dimsref.z;
		dimsref.z = 1;
	}
	tfloat* d_ref = ref.GetAsManagedDeviceTFloat();
	d_NormMonolithic(d_ref, d_ref, Elements(dimsref), T_NORM_MEAN01STD, nrefs);

	mxArrayAdapter ctf(prhs[2]);
	tfloat* d_ctf = ctf.GetAsManagedDeviceTFloat();

	mxArrayAdapter mask(prhs[3]);
	tfloat* d_mask = mask.GetAsManagedDeviceTFloat();

	mxArrayAdapter angularstep(prhs[4]);
	tfloat* h_angularstep = angularstep.GetAsManagedTFloat();

	Picker picker;
	picker.Initialize(d_ref, dimsref, d_mask, dimsimage);

	tfloat* d_bestccf = CudaMallocValueFilled(Elements(dimsimage), (tfloat)-99999);
	tfloat3* d_bestangle = (tfloat3*)CudaMallocValueFilled(Elements(dimsimage) * 3, (tfloat)0);

	picker.SetImage(d_image, d_ctf);
	for (int n = 0; n < nrefs; n++)
		picker.PerformCorrelation(h_angularstep[0], d_bestccf, d_bestangle);

	mwSize outputdims[3];
	outputdims[0] = dimsimage.x;
	outputdims[1] = dimsimage.y;
	outputdims[2] = dimsimage.z;
	mxArrayAdapter A(mxCreateNumericArray(3,
		outputdims,
		mxSINGLE_CLASS,
		mxREAL));
	A.SetFromDeviceTFloat(d_bestccf);
	plhs[0] = A.underlyingarray;


	cudaFree(d_bestccf);
	cudaFree(d_bestangle);
}