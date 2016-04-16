#include "..\Prerequisites.h"


void mexFunction(int nlhs, mxArray *plhs[],
	int nrhs, mxArray const *prhs[])
{
	char const * const errId = "GTOM:Pick:InvalidInput";

	mxInitGPU();

	if (nrhs != 1)
		mexErrMsgIdAndTxt(errId, "Wrong parameter count (1 expected).");

	mxArrayAdapter image(prhs[0]);
	int3 dimsimage = MWDimsToInt3(mxGetNumberOfDimensions(image.underlyingarray), mxGetDimensions(image.underlyingarray));
	tfloat* h_image = image.GetAsManagedTFloat();

	int3 center = toInt3(dimsimage.x / 2, dimsimage.y / 2, dimsimage.z / 2);
	tfloat* h_bins = MallocValueFilled(dimsimage.x / 2, (tfloat)0);
	int* h_samples = MallocValueFilled(dimsimage.x / 2, 0);

	for (int z = 0; z < dimsimage.z; z++)
	{
		int zz = z - center.z;
		for (int y = 0; y < dimsimage.y; y++)
		{
			int yy = y - center.y;
			for (int x = 0; x < dimsimage.x; x++)
			{
				int xx = x - center.x;

				tfloat val = *h_image++;
				int bin = (int)sqrt(zz * zz + yy * yy + xx * xx);
				if (bin >= dimsimage.x / 2)
					continue;

				h_bins[bin] += val;
				h_samples[bin]++;
			}
		}
	}

	for (int i = 0; i < dimsimage.x / 2; i++)
		h_bins[i] /= (tfloat)h_samples[i];

	mwSize outputdims[3];
	outputdims[0] = dimsimage.x / 2;
	outputdims[1] = 1;
	outputdims[2] = 1;
	mxArrayAdapter A(mxCreateNumericArray(3,
		outputdims,
		mxSINGLE_CLASS,
		mxREAL));
	A.SetFromTFloat(h_bins);
	plhs[0] = A.underlyingarray;

	free(h_bins);
	free(h_samples);
}