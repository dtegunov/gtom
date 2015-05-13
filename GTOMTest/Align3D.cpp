#include "Prerequisites.h"

TEST(Alignment, Align3D)
{
	cudaDeviceReset();

	//Case 1:
	{

		int3 dims = {32, 32, 32};
		int nvolumes = 100;
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

		tfloat* h_simmatrix = (tfloat*)malloc(nvolumes * nvolumes * sizeof(tfloat));
		tfloat* h_samplesmatrix = (tfloat*)malloc(nvolumes * nvolumes * sizeof(tfloat));

		for (int i = 0; i < nvolumes - 1; i++)
		{
			int elementsoffset = i + 1;
			int rowelements = nvolumes - elementsoffset;

			d_Align3D(d_volumes + Elements(dims) * elementsoffset, d_volumespsf + ElementsFFT(dims) * elementsoffset,
				d_volumes + Elements(dims) * i, d_volumespsf + ElementsFFT(dims) * i,
				NULL,
				dims,
				rowelements,
				ToRad(30.0f),
				tfloat2(0, PI2), tfloat2(ToRad(0.0f), ToRad(180.0f)), tfloat2(0, PI2),
				true,
				h_results + elementsoffset);

			for (int j = elementsoffset; j < nvolumes; j++)
			{
				h_simmatrix[j * nvolumes + i] = h_results[j].score;
				h_simmatrix[i * nvolumes + j] = h_results[j].score;

				h_samplesmatrix[j * nvolumes + i] = h_results[j].samples;
				h_samplesmatrix[i * nvolumes + j] = h_results[j].samples;
			}
			h_simmatrix[i * nvolumes + i] = 0;
			h_samplesmatrix[i * nvolumes + i] = 0;

			cout << i;
		}

		cudaFree(d_volumes);
		cudaFree(d_target);
		cudaFree(d_volumespsf);
		cudaFree(d_targetpsf);

		WriteToBinaryFile("d_simmatrix.bin", h_simmatrix, nvolumes * nvolumes * sizeof(tfloat));
		WriteToBinaryFile("d_samplesmatrix.bin", h_samplesmatrix, nvolumes * nvolumes * sizeof(tfloat));

		vector<float> diffs;
		for (int i = 0; i < nvolumes; i++)
			diffs.push_back(EulerCompare(tfloat3(ToRad((tfloat)(i + 25)), ToRad((tfloat)(i + 25)), ToRad((tfloat)(i + 25))), h_results[i].rotation));
		cout << diffs[0];

		free(h_results);
	}

	cudaDeviceReset();
}