#include "Prerequisites.h"

TEST(Relion, Project)
{
	cudaDeviceReset();

	//Case 1:
	{
		int3 dimsvolume = toInt3(283, 283, 283);
		int3 dimsproj = toInt3(280, 280, 1);
		int3 dimsproj2 = toInt3(6, 6, 1);
		uint nproj = 64;

		tfloat* h_volRe = (tfloat*)malloc(ElementsFFT(dimsvolume) * sizeof(tfloat));
		tfloat* h_volIm = (tfloat*)malloc(ElementsFFT(dimsvolume) * sizeof(tfloat));

		for (int i = 0; i < ElementsFFT(dimsvolume); i++)
		{
			h_volRe[i] = i;
			h_volIm[i] = -i;
		}

		tfloat* d_volRe = (tfloat*)CudaMallocFromHostArray(h_volRe, ElementsFFT(dimsvolume) * sizeof(tfloat));
		tfloat* d_volIm = (tfloat*)CudaMallocFromHostArray(h_volIm, ElementsFFT(dimsvolume) * sizeof(tfloat));

		cudaArray_t aRe, aIm;
		cudaTex tRe, tIm;
		d_BindTextureTo3DArray(d_volRe, aRe, tRe, toInt3FFT(dimsvolume), cudaFilterModeLinear, false);
		d_BindTextureTo3DArray(d_volIm, aIm, tIm, toInt3FFT(dimsvolume), cudaFilterModeLinear, false);

		tcomplex* d_proj = (tcomplex*)CudaMallocValueFilled(ElementsFFT(dimsproj) * 2 * nproj, (tfloat)0);
		tcomplex* d_proj2 = (tcomplex*)CudaMallocValueFilled(ElementsFFT(dimsproj2) * 2 * nproj, (tfloat)0);

		glm::mat3 matrix = Matrix3Euler(tfloat3(0.0f, 0.0f, ToRad(45.0f)));
		glm::mat3* matrices = (glm::mat3*)malloc(nproj * sizeof(glm::mat3));
		for (uint n = 0; n < nproj; n++)
			matrices[n] = Matrix3Euler(tfloat3(0.0f, n, -n));
		glm::mat3* d_matrices = (glm::mat3*)CudaMallocFromHostArray(matrices, nproj * sizeof(glm::mat3));

		d_rlnProject(tRe, tIm, dimsvolume, d_proj, dimsproj, dimsproj.x / 2, d_matrices, nproj);
		//d_rlnProject(tRe, tIm, dimsvolume, d_proj2, dimsproj2, dimsproj2.x / 2, 1, matrices, nproj);

		d_ConvertTComplexToSplitComplex(d_proj, d_volRe, d_volIm, ElementsFFT(dimsproj) * nproj);
		d_WriteMRC(d_volRe, toInt3(dimsproj.x / 2 + 1, dimsproj.y, nproj), "d_projRe.mrc");
		d_WriteMRC(d_volIm, toInt3(dimsproj.x / 2 + 1, dimsproj.y, nproj), "d_projIm.mrc");

		/*d_ConvertTComplexToSplitComplex(d_proj2, d_volRe, d_volIm, ElementsFFT(dimsproj2));
		d_WriteMRC(d_volRe, toInt3FFT(dimsproj2), "d_proj2Re.mrc");
		d_WriteMRC(d_volIm, toInt3FFT(dimsproj2), "d_proj2Im.mrc");*/
	}

	cudaDeviceReset();
}
