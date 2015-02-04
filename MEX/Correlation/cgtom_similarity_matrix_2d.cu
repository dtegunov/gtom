#include "..\Prerequisites.h"


void mexFunction(int nlhs, mxArray *plhs[],
	int nrhs, mxArray const *prhs[])
{
	char const * const errId = "GTOM:Correlation:SimilarityMatrix2D:InvalidInput";

	mxInitGPU();

	if (nrhs != 2)
		mexErrMsgIdAndTxt(errId, "Wrong parameter count (2 expected).");

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

	mxArrayAdapter anglesteps(prhs[1]);
	tfloat* h_anglesteps = anglesteps.GetAsManagedTFloat();
	
	tfloat* d_similarity = CudaMallocValueFilled(nimages, (tfloat)0);
	tfloat* h_simmatrix = MallocValueFilled(nimages * nimages, (tfloat)0);

	for (int i = 0; i < nimages - 1; i++)
	{
		d_SimilarityMatrixRow(d_images, d_imagesft, toInt2(dimsimage), nimages, h_anglesteps[0] + 0.5f, i, d_similarity);

		int rowelements = nimages - i - 1;
		int elementsoffset = i + 1;
		cudaMemcpy(h_simmatrix + nimages * i + elementsoffset, d_similarity + elementsoffset, rowelements * sizeof(tfloat), cudaMemcpyDeviceToHost);

		tfloat normfactor = (tfloat)1 / (tfloat)((long)Elements2(dimsimage) * (long)Elements2(dimsimage));
		for (int j = elementsoffset; j < nimages; j++)
			h_simmatrix[nimages * i + j] *= normfactor;

		mexPrintf("%d\n", i);
		mexEvalString("drawnow;");
	}

	cudaFree(d_similarity);
	cudaFree(d_imagesft);

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