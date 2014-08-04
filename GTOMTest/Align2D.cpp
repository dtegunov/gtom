#include "Prerequisites.h"

TEST(Alignment, Align2D)
{
	cudaDeviceReset();

	//Case 1:
	{
		int3 dims = {128, 128, 1};
		int numdata = 10;
		int numtargets = 10;
		tfloat* d_inputdata = (tfloat*)CudaMallocFromBinaryFile("Data\\Alignment\\Input_Align2DData_1.bin");
		tfloat* d_inputtargets = (tfloat*)CudaMallocFromBinaryFile("Data\\Alignment\\Input_Align2DTargets_1.bin");
		tfloat* h_desiredparams = (tfloat*)MallocFromBinaryFile("Data\\Alignment\\Input_Align2DParams_1.bin");

		tfloat3* h_outputparams = (tfloat3*)MallocValueFilled(numdata * 3, (tfloat)0);
		int* h_outputmembership = MallocValueFilled(numdata, 0);
		tfloat* h_outputscores = MallocValueFilled(numdata * numtargets, (tfloat)0);

		d_Align2D(d_inputdata, d_inputtargets, dims, numtargets, h_outputparams, h_outputmembership, h_outputscores, 24, ToRad(15), 3, T_ALIGN_MODE::T_ALIGN_BOTH, numdata);

		free(h_outputscores);
		free(h_outputmembership);
		free(h_outputparams);
		cudaFree(d_inputtargets);
		cudaFree(d_inputdata);
		free(h_desiredparams);
	}

	cudaDeviceReset();
}