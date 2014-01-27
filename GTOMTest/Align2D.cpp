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

		tfloat3* d_outputparams = (tfloat3*)CudaMallocValueFilled(numdata * 3, (tfloat)0);
		int* d_outputmembership = CudaMallocValueFilled(numdata, 0);
		tfloat* d_outputscores = CudaMallocValueFilled(numdata * numtargets, (tfloat)0);

		d_Align2D(d_inputdata, d_inputtargets, dims, numtargets, d_outputparams, d_outputmembership, d_outputscores, 24, ToRad(10), 3, T_ALIGN_MODE::T_ALIGN_BOTH, numdata);

		cudaFree(d_outputscores);
		cudaFree(d_outputparams);
		cudaFree(d_inputtargets);
		cudaFree(d_inputdata);
		free(h_desiredparams);
	}

	cudaDeviceReset();
}