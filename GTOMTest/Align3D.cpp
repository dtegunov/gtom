#include "Prerequisites.h"
#include "..\GLMFunctions.cuh"

TEST(Alignment, Align3D)
{
	cudaDeviceReset();

	//Case 1:
	{

		int3 dims = {128, 128, 128};
		tfloat* d_inputdata = (tfloat*)CudaMallocFromBinaryFile("Data\\Alignment\\Input_Align3DData_1.bin");
		tfloat* d_target = (tfloat*)CudaMallocFromBinaryFile("Data\\Alignment\\Input_Align3DTarget_1.bin");
		
		tfloat3 position = tfloat3(0);
		tfloat3 rotation = tfloat3(0);
		int membership = -1;
		tfloat score = (tfloat)-1;

		d_Align3D(d_inputdata, d_target, dims, 1, position, rotation, &membership, &score, &position, &rotation, 0, tfloat3(ToRad(180.0f), ToRad(180.0f), ToRad(180.0f)), ToRad(10.0f), 3, T_ALIGN_ROT);

		rotation = tfloat3(ToDeg(rotation.x), ToDeg(rotation.y), ToDeg(rotation.z));

		cudaFree(d_target);
		cudaFree(d_inputdata);
	}

	cudaDeviceReset();
}