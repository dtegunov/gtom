#include "..\Prerequisites.h"


void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
    char const * const errId = "GTOM:Generics:Norm:InvalidInput";
    char const * const errMsg = "Invalid input to MEX file.";

    mxInitGPU();

	if (nrhs < 2 || (!mxIsSingle(prhs[0]) && !mxIsDouble(prhs[0])))
        mexErrMsgIdAndTxt(errId, errMsg);

	mxArrayAdapter A(prhs[0]);
	int ndims = mxGetNumberOfDimensions(A.underlyingarray);
	int3 dimensions = MWDimsToInt3(ndims, mxGetDimensions(A.underlyingarray));

	T_NORM_MODE mode;
	tfloat scf = 0;
    if(mxIsChar(prhs[1]))
	{
		char* modestring = mxArrayToString(prhs[1]);
		if(strcmp(modestring, "phase") == 0)
			mode = T_NORM_MODE::T_NORM_PHASE;
		else if(strcmp(modestring, "std1") == 0)
			mode = T_NORM_MODE::T_NORM_STD1;
		else if(strcmp(modestring, "std2") == 0)
			mode = T_NORM_MODE::T_NORM_STD2;
		else if(strcmp(modestring, "std3") == 0)
			mode = T_NORM_MODE::T_NORM_STD3;
		else if(strcmp(modestring, "mean0+1std") == 0)
			mode = T_NORM_MODE::T_NORM_MEAN01STD;
		else if(strcmp(modestring, "oscar") == 0)
			mode = T_NORM_MODE::T_NORM_OSCAR;
		else
			mexErrMsgIdAndTxt(errId, "Invalid mode string.");
	}
	else 
	{
		mode = T_NORM_MODE::T_NORM_CUSTOM;
		if(mxIsSingle(prhs[1]))
			scf = (tfloat)((float*)mxGetPr(prhs[1]))[0];
		else if(mxIsDouble(prhs[1]))
			scf = (tfloat)mxGetPr(prhs[1])[0];
		else
			mexErrMsgIdAndTxt(errId, "Invalid custom StdDev.");
	}

	tfloat* d_input = A.GetAsManagedDeviceTFloat();
		
	d_Norm(d_input, 
		   d_input, 
		   (size_t)dimensions.x * (size_t)dimensions.y * (size_t)dimensions.z,
		   (int*)NULL,
		   mode,
		   scf,
		   1);
	
	mxArrayAdapter B(mxCreateNumericArray(mxGetNumberOfDimensions(A.underlyingarray),
					 mxGetDimensions(A.underlyingarray),
					 mxGetClassID(A.underlyingarray),
					 mxREAL));
	B.SetFromDeviceTFloat(d_input);
	plhs[0] = B.underlyingarray;
}