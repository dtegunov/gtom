#include "Prerequisites.h"

TEST(Alignment, Align3D)
{
	cudaDeviceReset();

	//Case 1:
	{

		int3 dims = {32, 32, 32};
		int nvolumes = 51;
		tfloat* d_volumes = (tfloat*)CudaMallocFromBinaryFile("Data\\Alignment\\Input_Align3DData_1.bin");
		tfloat* d_target = (tfloat*)CudaMallocFromBinaryFile("Data\\Alignment\\Input_Align3DTarget_1.bin");
		
		tfloat* d_volumespsf = (tfloat*)CudaMallocFromBinaryFile("Data\\Alignment\\Input_Align3DDataPSF_1.bin");
		tfloat* d_targetpsf = (tfloat*)CudaMallocFromBinaryFile("Data\\Alignment\\Input_Align3DTargetPSF_1.bin");

		Align3DParams* h_results = (Align3DParams*)malloc(nvolumes * sizeof(Align3DParams));

		d_NormMonolithic(d_volumes, d_volumes, Elements(dims), T_NORM_MEAN01STD, nvolumes);
		d_NormMonolithic(d_target, d_target, Elements(dims), T_NORM_MEAN01STD, 1);

		d_HannMask(d_volumes, d_volumes, dims, NULL, NULL, nvolumes);
		d_HannMask(d_target, d_target, dims, NULL, NULL, 1);

		d_RemapFull2FullFFT(d_volumes, d_volumes, dims, nvolumes);
		d_RemapFull2FullFFT(d_target, d_target, dims, 1);

		d_Align3D(d_volumes, d_volumespsf,
			d_target, d_targetpsf,
			NULL,
			dims,
			nvolumes,
			ToRad(10.0f),
			tfloat2(0, PI2), tfloat2(0, PIHALF), tfloat2(0, PI2),
			true,
			h_results);

		vector<float> diffs;
		for (int i = 0; i < nvolumes; i++)
			diffs.push_back(EulerCompare(tfloat3(ToRad((tfloat)-i), ToRad((tfloat)i), ToRad((tfloat)i * 2.0f)), h_results[i].rotation));
		cout << diffs[0];

		free(h_results);
	}

	cudaDeviceReset();
}