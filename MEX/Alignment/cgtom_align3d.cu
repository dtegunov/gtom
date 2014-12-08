#include "..\Prerequisites.h"

//RHS:
//	Volume
//	Target volume
//	Maximum shift
//	Maximum rotation
//	Rotation step
//	Rotation refinements
//	Alignment mode

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
    char const * const errId = "GTOM:Alignment:Align3D:InvalidInput";
    char const * const errMsg = "Invalid input to MEX file.";

    mxInitGPU();

	if (nrhs < 3)
        mexErrMsgIdAndTxt(errId, errMsg);

	mxArrayAdapter vol(prhs[0]);
	int ndims = mxGetNumberOfDimensions(vol.underlyingarray);
	int3 dimsvol = MWDimsToInt3(ndims, mxGetDimensions(vol.underlyingarray));
	tfloat* d_vol = vol.GetAsManagedDeviceTFloat();

	mxArrayAdapter targetvol(prhs[1]);
	tfloat* d_targetvol = targetvol.GetAsManagedDeviceTFloat();

	tfloat maxshift = (tfloat)((double*)mxGetData(prhs[2]))[0];
	tfloat maxrotationX = (tfloat)((double*)mxGetData(prhs[3]))[0];
	tfloat maxrotationY = (tfloat)((double*)mxGetData(prhs[3]))[1];
	tfloat maxrotationZ = (tfloat)((double*)mxGetData(prhs[3]))[2];
	tfloat rotationstep = (tfloat)((double*)mxGetData(prhs[4]))[0];
	tfloat rotationrefinements = (tfloat)((double*)mxGetData(prhs[5]))[0];
	T_ALIGN_MODE alignmentmode = (int)((double*)mxGetData(prhs[5]))[0] == 0 ? T_ALIGN_ROT : T_ALIGN_BOTH;

	tfloat3 shift = tfloat3(0);
	tfloat3 rotation = tfloat3(0);
	int membership = 0;
	tfloat score = 0;
	d_Align3D(d_vol, d_targetvol, dimsvol, 1, shift, rotation, &membership, &score, &shift, &rotation, maxshift, tfloat3(maxrotationX, maxrotationY, maxrotationZ), rotationstep, rotationrefinements, alignmentmode);
	
	mwSize outputdims[3];
	outputdims[0] = 6;
	outputdims[1] = 1;
	outputdims[2] = 1;
	mxArrayAdapter B(mxCreateNumericArray(3,
					 outputdims,
					 mxSINGLE_CLASS,
					 mxREAL));
	tfloat h_output[6];
	h_output[0] = shift.x;
	h_output[1] = shift.y;
	h_output[2] = shift.z;
	h_output[3] = rotation.x;
	h_output[4] = rotation.y;
	h_output[5] = rotation.z;
	B.SetFromTFloat(h_output);
	plhs[0] = B.underlyingarray;
}