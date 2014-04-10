#include "Prerequisites.h"

TEST(Transformation, Rotation)
{
	cudaDeviceReset();

	//Case 1:
	//{
	//	int3 dims = {8, 8, 8};
	//	tfloat3 angles = tfloat3(ToRad(0.0f), ToRad(180.0f), ToRad(0.0f));
	//	tfloat* d_input = (tfloat*)CudaMallocFromBinaryFile("Data\\Transformation\\Input_Rotate3D_1.bin");
	//	tfloat* desired_output = (tfloat*)MallocFromBinaryFile("Data\\Transformation\\Output_Rotate3D_1.bin");
	//	tfloat* d_output = CudaMallocValueFilled(Elements(dims), (tfloat)-1);

	//	d_Rotate3D(d_input, d_output, dims, &angles, T_INTERP_MODE::T_INTERP_CUBIC);
	//	tfloat* h_output = (tfloat*)MallocFromDeviceArray(d_output, Elements(dims) * sizeof(tfloat));
	//
	//	double MeanRelative = GetMeanRelativeError((tfloat*)desired_output, (tfloat*)h_output, Elements(dims));
	//	//ASSERT_LE(MeanRelative, 1e-5);

	//	cudaFree(d_input);
	//	cudaFree(d_output);
	//	free(desired_output);
	//	free(h_output);
	//}

	cudaDeviceReset();

	//Case 1:
	{
		int3 dims = {16, 16, 1};
		tfloat angle = ToRad(0);
		tfloat* d_input = (tfloat*)CudaMallocFromBinaryFile("Data\\Transformation\\Input_Rotate2D_1.bin");
		tfloat* desired_output = (tfloat*)MallocFromBinaryFile("Data\\Transformation\\Output_Rotate2D_1.bin");
		tfloat* d_output = CudaMallocValueFilled(Elements(dims), (tfloat)-1);

		d_RemapFull2FullFFT(d_input, d_input, dims);

		tcomplex* d_inputft;
		cudaMalloc((void**)&d_inputft, ElementsFFT(dims) * sizeof(tcomplex));
		d_FFTR2C(d_input, d_inputft, 2, dims);
		d_RemapHalfFFT2Half(d_inputft, d_inputft, dims);

		d_Rotate2DFT(d_inputft, d_inputft, dims, angle, T_INTERP_LINEAR);
		d_RemapHalf2HalfFFT(d_inputft, d_inputft, dims);

		d_IFFTC2R(d_inputft, d_output, 2, dims);
		d_RemapFullFFT2Full(d_output, d_output, dims);

		tfloat* h_output = (tfloat*)MallocFromDeviceArray(d_output, Elements(dims) * sizeof(tfloat));
	
		double MeanRelative = GetMeanRelativeError((tfloat*)desired_output, (tfloat*)h_output, Elements(dims));
		//ASSERT_LE(MeanRelative, 1e-5);

		cudaFree(d_input);
		cudaFree(d_output);
		free(desired_output);
		free(h_output);
	}

	cudaDeviceReset();
}