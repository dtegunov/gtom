#include "Prerequisites.h"

TEST(Correlation, SimilarityMatrix)
{
	cudaDeviceReset();

	//Case 1:
	{
		/*int2 dimsimage = toInt2(32, 32);
		int nimages = 3;
		tfloat* d_input = (tfloat*)CudaMallocFromBinaryFile("Data\\Correlation\\Input_SimMatrix.bin");

		tfloat* d_inputnorm;
		cudaMalloc((void**)&d_inputnorm, Elements2(dimsimage) * nimages * sizeof(tfloat));
		d_HannMask(d_input, d_inputnorm, toInt3(dimsimage), NULL, NULL, nimages);
		d_NormMonolithic(d_inputnorm, d_inputnorm, Elements2(dimsimage), T_NORM_MEAN01STD, nimages);

		CudaWriteToBinaryFile("d_inputnorm.bin", d_inputnorm, Elements2(dimsimage) * nimages * sizeof(tfloat));
		
		tcomplex* d_inputft;
		cudaMalloc((void**)&d_inputft, ElementsFFT2(dimsimage) * nimages * sizeof(tcomplex));
		d_FFTR2C(d_inputnorm, d_inputft, 2, toInt3(dimsimage), nimages);

		tfloat* d_similarity = CudaMallocValueFilled(nimages, (tfloat)0);

		d_SimilarityMatrixRow(d_input, d_inputft, dimsimage, nimages, 360, 0, d_similarity);

		tfloat* h_similarity = (tfloat*)MallocFromDeviceArray(d_similarity, nimages * sizeof(tfloat));
		free(h_similarity);*/
	}

	//Case 2:
	{
		int2 dimsimage = toInt2(32, 32);
		int nimages = 2;
		tfloat* d_input = (tfloat*)CudaMallocFromBinaryFile("Data\\Correlation\\Input_SimMatrix.bin");

		tfloat* d_inputnorm;
		cudaMalloc((void**)&d_inputnorm, Elements2(dimsimage) * nimages * sizeof(tfloat));
		d_HannMask(d_input, d_inputnorm, toInt3(dimsimage), NULL, NULL, nimages);
		d_NormMonolithic(d_inputnorm, d_inputnorm, Elements2(dimsimage), T_NORM_MEAN01STD, nimages);

		tcomplex* d_inputft;
		cudaMalloc((void**)&d_inputft, ElementsFFT2(dimsimage) * nimages * sizeof(tcomplex));
		d_FFTR2C(d_inputnorm, d_inputft, 2, toInt3(dimsimage), nimages);

		int linewidth = 1;
		int anglesteps = 90;
		int2 dimsline = toInt2(dimsimage.x, linewidth);
		int2 dimslines = toInt2(dimsline.x, dimsline.y * anglesteps);
		tcomplex* d_linesft;
		cudaMalloc((void**)&d_linesft, ElementsFFT2(dimslines) * nimages * sizeof(tcomplex));
		tfloat* d_lines;
		cudaMalloc((void**)&d_lines, Elements2(dimslines) * nimages * sizeof(tfloat));

		d_FFTLines(d_inputft, d_linesft, dimsimage, T_INTERP_CUBIC, anglesteps, linewidth, nimages);
		d_IFFTC2R(d_linesft, d_lines, DimensionCount(toInt3(dimsline)), toInt3(dimsline), anglesteps * nimages, false);

		d_NormMonolithic(d_lines, d_lines, Elements2(dimsline), T_NORM_MEAN01STD, anglesteps * nimages);
		CudaWriteToBinaryFile("d_lines.bin", d_lines, Elements2(dimslines) * nimages * sizeof(tfloat));
		d_FFTR2C(d_lines, d_linesft, DimensionCount(toInt3(dimsline)), toInt3(dimsline), anglesteps * nimages);

		tfloat* d_similarity = CudaMallocValueFilled(nimages, (tfloat)0);

		d_LineSimilarityMatrixRow(d_linesft, dimsimage, nimages, linewidth, anglesteps, 0, d_similarity);

		tfloat* h_similarity = (tfloat*)MallocFromDeviceArray(d_similarity, nimages * sizeof(tfloat));
		free(h_similarity);
	}

	cudaDeviceReset();
}