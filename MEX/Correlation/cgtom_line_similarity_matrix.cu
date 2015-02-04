#include "..\Prerequisites.h"


void mexFunction(int nlhs, mxArray *plhs[],
	int nrhs, mxArray const *prhs[])
{
	char const * const errId = "GTOM:Correlation:LineSimilarityMatrix:InvalidInput";

	mxInitGPU();

	if (nrhs != 3)
		mexErrMsgIdAndTxt(errId, "Wrong parameter count (3 expected).");

	mxArrayAdapter images(prhs[0]);
	int3 dimsimage = MWDimsToInt3(mxGetNumberOfDimensions(images.underlyingarray), mxGetDimensions(images.underlyingarray));
	int nimages = dimsimage.z;
	dimsimage.z = 1;
	tfloat* d_images = images.GetAsManagedDeviceTFloat();

	tfloat* d_imagesmasked;
	cudaMalloc((void**)&d_imagesmasked, Elements2(dimsimage) * nimages * sizeof(tfloat));
	d_HannMask(d_images, d_imagesmasked, dimsimage, NULL, NULL, nimages);
	d_NormMonolithic(d_imagesmasked, d_imagesmasked, Elements2(dimsimage), T_NORM_MEAN01STD, nimages);

	tcomplex* d_imagesft;
	cudaMalloc((void**)&d_imagesft, ElementsFFT2(dimsimage) * nimages * sizeof(tcomplex));
	d_FFTR2C(d_imagesmasked, d_imagesft, 2, dimsimage, nimages);
	cudaFree(d_imagesmasked);

	mxArrayAdapter a_anglesteps(prhs[1]);
	tfloat* h_anglesteps = a_anglesteps.GetAsManagedTFloat();
	int anglesteps = (int)(h_anglesteps[0] + 0.5);

	mxArrayAdapter a_linewidth(prhs[2]);
	tfloat* h_linewidth = a_linewidth.GetAsManagedTFloat();
	int linewidth = (int)(h_linewidth[0] + 0.5);

	int2 dimsline = toInt2(dimsimage.x, linewidth);
	int2 dimslines = toInt2(dimsline.x, dimsline.y * anglesteps);
	tcomplex* d_linesft;
	cudaMalloc((void**)&d_linesft, ElementsFFT2(dimslines) * nimages * sizeof(tcomplex));

	d_FFTLines(d_imagesft, d_linesft, toInt2(dimsimage), T_INTERP_CUBIC, anglesteps, linewidth, nimages);
	cudaFree(d_imagesft);

	tfloat* d_lines;
	cudaMalloc((void**)&d_lines, Elements2(dimslines) * nimages * sizeof(tfloat));
	d_IFFTC2R(d_linesft, d_lines, DimensionCount(toInt3(dimsline)), toInt3(dimsline), anglesteps * nimages, false);

	d_NormMonolithic(d_lines, d_lines, Elements2(dimsline), T_NORM_MEAN01STD, anglesteps * nimages);
	d_FFTR2C(d_lines, d_linesft, DimensionCount(toInt3(dimsline)), toInt3(dimsline), anglesteps * nimages);
	cudaFree(d_lines);

	tfloat* d_similarity = CudaMallocValueFilled(nimages, (tfloat)0);
	tfloat* h_simmatrix = MallocValueFilled(nimages * nimages, (tfloat)0);

	for (int i = 0; i < nimages - 1; i++)
	{
		d_LineSimilarityMatrixRow(d_linesft, toInt2(dimsimage), nimages, linewidth, anglesteps, i, d_similarity);

		int rowelements = nimages - i - 1;
		int elementsoffset = i + 1;
		cudaMemcpy(h_simmatrix + nimages * i + elementsoffset, d_similarity + elementsoffset, rowelements * sizeof(tfloat), cudaMemcpyDeviceToHost);

		tfloat normfactor = (tfloat)1 / (tfloat)((long)Elements2(dimsline) * (long)Elements2(dimsline));
		for (int j = elementsoffset; j < nimages; j++)
			h_simmatrix[nimages * i + j] *= normfactor;

		mexPrintf("%d\n", i);
		mexEvalString("drawnow;");
	}

	cudaFree(d_similarity);
	cudaFree(d_linesft);

	mwSize outputdims[3];
	outputdims[0] = nimages;
	outputdims[1] = nimages;
	outputdims[2] = 1;
	mxArrayAdapter A(mxCreateNumericArray(2,
		outputdims,
		mxGetClassID(images.underlyingarray),
		mxREAL));
	A.SetFromTFloat(h_simmatrix);
	plhs[0] = A.underlyingarray;

	free(h_simmatrix);
}