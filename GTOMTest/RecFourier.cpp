#include "Prerequisites.h"

TEST(Reconstruction, Fourier)
{
	cudaDeviceReset();

	//Case 1:
	/*{
		int3 dimsvolume = {16, 16, 16};
		int3 dimsimage = {16, 16, 2249};

		tfloat* d_inputproj = (tfloat*)CudaMallocFromBinaryFile("Data\\Reconstruction\\Input_ARTProj_2.bin");
		tfloat3* h_inputangles = (tfloat3*)MallocFromBinaryFile("Data\\Reconstruction\\Input_ARTAngles_2.bin");
		tfloat* desired_output = (tfloat*)MallocFromBinaryFile("Data\\Reconstruction\\Output_ART_2.bin");

		tfloat* d_volume;
		cudaMalloc((void**)&d_volume, Elements(dimsvolume) * sizeof(tfloat));
		

		tfloat* h_output = (tfloat*)MallocFromDeviceArray(d_volume, Elements(dimsvolume) * sizeof(tfloat));
		tfloat outputmax = (tfloat)-999999;
		for (int i = 0; i < Elements(dimsvolume); i++)
			outputmax = max(outputmax, h_output[i]);
		//for (int i = 0; i < Elements(dimsvolume); i++)
			//h_output[i] /= outputmax;
	
		double MeanAbsolute = GetMeanAbsoluteError((tfloat*)desired_output, (tfloat*)h_output, Elements(dimsvolume));
		//ASSERT_LE(MeanRelative, 1e-5);

		cudaFree(d_volume);
		cudaFree(d_inputproj);
		free(desired_output);
		free(h_output);
	}*/

	//Case 2:
	{
		int3 dimsori = toInt3(128, 128, 128);
		int3 dimspadded = toInt3(259, 259, 259);

		tfloat* h_weights = (tfloat*)malloc(ElementsFFT(dimspadded) * sizeof(tfloat));
		ReadMRC("d_weights.mrc", (void**)&h_weights);
		tfloat* d_weights = (tfloat*)CudaMallocFromHostArray(h_weights, ElementsFFT(dimspadded) * sizeof(tfloat));

		tcomplex* h_data = (tcomplex*)malloc(ElementsFFT(dimspadded) * sizeof(tcomplex));
		ReadMRC("d_dataRe.mrc", (void**)&h_weights);
		for (int i = 0; i < ElementsFFT(dimspadded); i++)
			h_data[i].x = h_weights[i];
		ReadMRC("d_dataIm.mrc", (void**)&h_weights);
		for (int i = 0; i < ElementsFFT(dimspadded); i++)
			h_data[i].y = h_weights[i];
		tcomplex* d_data = (tcomplex*)CudaMallocFromHostArray(h_data, ElementsFFT(dimspadded) * sizeof(tcomplex));

		tfloat* d_reconstructed = CudaMallocValueFilled(Elements(dimsori), (tfloat)0);

		d_ReconstructGridding(d_data, d_weights, d_reconstructed, dimsori, dimspadded);
	}

	cudaDeviceReset();
}