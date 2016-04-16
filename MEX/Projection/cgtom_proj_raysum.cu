#include "..\Prerequisites.h"


void mexFunction(int nlhs, mxArray *plhs[],
	int nrhs, mxArray const *prhs[])
{
	char const * const errId = "GTOM:Projection:RaySum:InvalidInput";

	mxInitGPU();

	if (nrhs != 3)
		mexErrMsgIdAndTxt(errId, "Wrong parameter count (3 expected).");

	mxArrayAdapter volume(prhs[0]);
	int3 dimsvolume = MWDimsToInt3(mxGetNumberOfDimensions(volume.underlyingarray), mxGetDimensions(volume.underlyingarray));
	tfloat* d_volume = volume.GetAsManagedDeviceTFloat();

	cudaTex t_volume;
	cudaArray_t a_volume;
	d_BindTextureTo3DArray(d_volume, a_volume, t_volume, dimsvolume, cudaFilterModeLinear, false);

	mxArrayAdapter start(prhs[1]);
	int3 dimsstart = MWDimsToInt3(mxGetNumberOfDimensions(start.underlyingarray), mxGetDimensions(start.underlyingarray));
	if (dimsstart.x % 3 != 0)
		mexErrMsgIdAndTxt(errId, "3 values per start vector expected.");
	glm::vec3* d_start = (glm::vec3*)start.GetAsUnmanagedDeviceTFloat();
	
	mxArrayAdapter finish(prhs[2]);
	int3 dimsfinish = MWDimsToInt3(mxGetNumberOfDimensions(finish.underlyingarray), mxGetDimensions(finish.underlyingarray));
	if (dimsstart.x != dimsfinish.x || dimsstart.y != dimsfinish.y)
		mexErrMsgIdAndTxt(errId, "Number of start and finish vectors must be equal.");
	glm::vec3* d_finish = (glm::vec3*)finish.GetAsUnmanagedDeviceTFloat();

	uint batch = dimsstart.x / 3 * dimsstart.y;
	tfloat* d_sums;
	cudaMalloc((void**)&d_sums, batch * sizeof(tfloat));

	d_RaySum(t_volume, d_start, d_finish, d_sums, T_INTERP_LINEAR, 2, batch);

	cudaDestroyTextureObject(t_volume);
	cudaFreeArray(a_volume);

	mwSize outputdims[2];
	outputdims[0] = dimsstart.x / 3;
	outputdims[1] = dimsstart.y;
	mxArrayAdapter A(mxCreateNumericArray(2,
		outputdims,
		mxGetClassID(volume.underlyingarray),
		mxREAL));
	A.SetFromDeviceTFloat(d_sums);
	cudaFree(d_sums);
	plhs[0] = A.underlyingarray;
}