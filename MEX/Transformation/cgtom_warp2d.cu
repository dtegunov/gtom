#include "..\Prerequisites.h"


void mexFunction(int nlhs, mxArray *plhs[],
	int nrhs, mxArray const *prhs[])
{
	char const * const errId = "GTOM:Transform:Warp2D:InvalidInput";

	mxInitGPU();

	if (nrhs != 3)
		mexErrMsgIdAndTxt(errId, "Wrong parameter count (3 expected).");

	mxArrayAdapter image(prhs[0]);
	int3 dimsimage = MWDimsToInt3(mxGetNumberOfDimensions(image.underlyingarray), mxGetDimensions(image.underlyingarray));
	uint nimages = dimsimage.z;
	dimsimage.z = 1;
	tfloat* d_image = image.GetAsManagedDeviceTFloat();

	mxArrayAdapter gridx(prhs[1]);
	int3 dimsgridx = MWDimsToInt3(mxGetNumberOfDimensions(gridx.underlyingarray), mxGetDimensions(gridx.underlyingarray));
	if (dimsgridx.z != nimages)
		mexErrMsgIdAndTxt(errId, "Number of warp grids should match number of images.");
	tfloat* h_gridx = gridx.GetAsManagedTFloat();

	mxArrayAdapter gridy(prhs[2]);
	int3 dimsgridy = MWDimsToInt3(mxGetNumberOfDimensions(gridy.underlyingarray), mxGetDimensions(gridy.underlyingarray));
	if (dimsgridy.z != nimages)
		mexErrMsgIdAndTxt(errId, "Number of warp grids should match number of images.");
	tfloat* h_gridy = gridy.GetAsManagedTFloat();

	int2 dimsgrid = toInt2(dimsgridx);
	tfloat2* h_grid = (tfloat2*)malloc(Elements2(dimsgrid) * nimages * sizeof(tfloat2));
	for (uint i = 0; i < Elements2(dimsgrid) * nimages; i++)
		h_grid[i] = tfloat2(h_gridx[i], h_gridy[i]);
	tfloat2* d_grid = (tfloat2*)CudaMallocFromHostArray(h_grid, Elements2(dimsgrid) * nimages * sizeof(tfloat2));
	free(h_grid);

	tfloat* d_output;
	cudaMalloc((void**)&d_output, Elements2(dimsimage) * nimages * sizeof(tfloat));

	d_Warp2D(d_image, toInt2(dimsimage), d_grid, dimsgrid, d_output, nimages);

	mwSize outputdims[3];
	outputdims[0] = dimsimage.x;
	outputdims[1] = dimsimage.y;
	outputdims[2] = nimages;
	mxArrayAdapter A(mxCreateNumericArray(3,
		outputdims,
		mxSINGLE_CLASS,
		mxREAL));
	A.SetFromDeviceTFloat(d_output);
	plhs[0] = A.underlyingarray;
	cudaFree(d_output);
	cudaFree(d_grid);
}