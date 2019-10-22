#include "Prerequisites.cuh"
#include "Alignment.cuh"
#include "Angles.cuh"
#include "Correlation.cuh"
#include "CubicInterp.cuh"
#include "CTF.cuh"
#include "FFT.cuh"
#include "Generics.cuh"
#include "Helper.cuh"
#include "ImageManipulation.cuh"
#include "Masking.cuh"
#include "Optimization.cuh"
#include "Transformation.cuh"

namespace gtom
{
	class Align3DOptimizer : public Optimizer
	{
	private:
		int nvolumes;
		int3 dimsvolume;
		tfloat2 thetarange;

		tcomplex* d_targetftrotated;
		tfloat* d_volumespsfsphere;
		tfloat* d_buffer1;
		tfloat* d_buffer2;
		tfloat* d_buffer3;
		tfloat* d_samples;

		cudaTex t_targetftRe, t_targetftIm, t_targetpsf;
		cudaArray_t a_targetftRe, a_targetftIm, a_targetpsf;
		tcomplex* d_volumesft;
		tfloat* d_volumespsf;
		tfloat* d_mask;

		tfloat* d_scores;
		tfloat3* d_positions;

		tfloat* d_targetmask;
		tfloat* h_targetmasksum;

		cufftHandle planforw, planback;

	public:
		Align3DParams* h_results;
		bool usevolumemask;

		Align3DOptimizer(tcomplex* _d_volumesft, tfloat* _d_volumespsf,
			tcomplex* _d_targetft, tfloat* _d_targetpsf,
			tfloat* _d_targetmask,
			int3 _dimsvolume,
			int _nvolumes,
			tfloat2 _thetarange)
		{
			nvolumes = _nvolumes;
			dimsvolume = _dimsvolume;
			thetarange = _thetarange;
			d_volumesft = _d_volumesft;
			d_volumespsf = _d_volumespsf;

			cudaMalloc((void**)&d_targetftrotated, ElementsFFT(dimsvolume) * nvolumes * sizeof(tcomplex));
			cudaMalloc((void**)&d_volumespsfsphere, ElementsFFT(dimsvolume) * nvolumes * sizeof(tfloat));
			cudaMalloc((void**)&d_buffer1, ElementsFFT(dimsvolume) * nvolumes * sizeof(tcomplex));
			cudaMalloc((void**)&d_buffer2, ElementsFFT(dimsvolume) * nvolumes * sizeof(tcomplex));
			cudaMalloc((void**)&d_buffer3, ElementsFFT(dimsvolume) * nvolumes * sizeof(tcomplex));
			cudaMalloc((void**)&d_samples, nvolumes * sizeof(tfloat));

			cudaMalloc((void**)&d_positions, nvolumes * sizeof(tfloat3));
			cudaMalloc((void**)&d_scores, nvolumes * sizeof(tfloat));

			{
				tfloat* d_tempRe, *d_tempIm;
				cudaMalloc((void**)&d_tempRe, ElementsFFT(dimsvolume) * sizeof(tfloat));
				cudaMalloc((void**)&d_tempIm, ElementsFFT(dimsvolume) * sizeof(tfloat));

				int3 dimsfft = toInt3(dimsvolume.x / 2 + 1, dimsvolume.y, dimsvolume.z);

				d_ConvertTComplexToSplitComplex(_d_targetft, d_tempRe, d_tempIm, ElementsFFT(dimsvolume));
				d_CubicBSplinePrefilter3D(d_tempRe, dimsfft);
				d_CubicBSplinePrefilter3D(d_tempIm, dimsfft);
				d_BindTextureTo3DArray(d_tempRe, a_targetftRe, t_targetftRe, dimsfft, cudaFilterModeLinear, false);
				d_BindTextureTo3DArray(d_tempIm, a_targetftIm, t_targetftIm, dimsfft, cudaFilterModeLinear, false);

				cudaMemcpy(d_tempRe, _d_targetpsf, ElementsFFT(dimsvolume) * sizeof(tfloat), cudaMemcpyDeviceToDevice);
				d_CubicBSplinePrefilter3D(d_tempRe, dimsfft);
				d_BindTextureTo3DArray(d_tempRe, a_targetpsf, t_targetpsf, dimsfft, cudaFilterModeLinear, false);

				cudaFree(d_tempRe);
				cudaFree(d_tempIm);
			}

			h_results = (Align3DParams*)malloc(nvolumes * sizeof(Align3DParams));
			usevolumemask = false;

			d_mask = CudaMallocValueFilled(Elements(dimsvolume), (tfloat)1);
			d_SphereMask(d_mask, d_mask, dimsvolume, NULL, 0, NULL, false);
			d_RemapFull2HalfFFT(d_mask, d_mask, dimsvolume);
			d_RemapHalf2HalfFFT(d_volumespsf, d_buffer3, dimsvolume, nvolumes);
			d_MultiplyByVector(d_buffer3, d_mask, d_volumespsfsphere, ElementsFFT(dimsvolume), nvolumes);

			d_targetmask = _d_targetmask;
			if (d_targetmask != NULL)
			{
				tfloat* d_result = CudaMallocValueFilled(nvolumes, (tfloat)0);
				d_Sum(d_targetmask, d_result, Elements(dimsvolume), nvolumes);
				cudaMemcpy(h_targetmasksum, d_result, nvolumes * sizeof(tfloat), cudaMemcpyDeviceToHost);
				cudaFree(d_result);
			}

			planforw = d_FFTR2CGetPlan(3, dimsvolume, nvolumes);
			planback = d_IFFTC2RGetPlan(3, dimsvolume, nvolumes);
		}

		~Align3DOptimizer()
		{
			free(h_results);

			cudaFree(d_mask);
			cudaFree(d_samples);
			cudaFree(d_scores);
			cudaFree(d_positions);
			cudaFree(d_buffer3);
			cudaFree(d_buffer2);
			cudaFree(d_buffer1);
			cudaFree(d_volumespsfsphere);
			cudaFree(d_targetftrotated);

			cufftDestroy(planforw);
			cufftDestroy(planback);

			cudaDestroyTextureObject(t_targetftRe);
			cudaDestroyTextureObject(t_targetftIm);
			cudaDestroyTextureObject(t_targetpsf);
			cudaFreeArray(a_targetftRe);
			cudaFreeArray(a_targetftIm);
			cudaFreeArray(a_targetpsf);
		}

		column_vector ComputeScores(column_vector& vec)
		{
			////////////////////
			// Rotate volumes://
			////////////////////

			tfloat3* h_translations = (tfloat3*)malloc(nvolumes * sizeof(tfloat3));
			tfloat3* h_rotations = (tfloat3*)malloc(nvolumes * sizeof(tfloat3));
			bool equalangles = true;
			tfloat3 compareto = tfloat3((tfloat)vec(0), (tfloat)vec(1), (tfloat)vec(2));
			for (int i = 0; i < nvolumes; i++)
			{
				if (!usevolumemask)
					h_rotations[i] = tfloat3((tfloat)vec(i * 3), (tfloat)vec(i * 3 + 1), (tfloat)vec(i * 3 + 2));
				else
				{
					h_rotations[i] = tfloat3((tfloat)vec(i * 6), (tfloat)vec(i * 6 + 1), (tfloat)vec(i * 6 + 2));
					h_translations[i] = tfloat3((tfloat)(vec(i * 6 + 3) * 30.0), (tfloat)(vec(i * 6 + 4) * 30.0), (tfloat)(vec(i * 6 + 5) * 30.0));
				}
				h_rotations[i].y = min(thetarange.y, max(thetarange.x, h_rotations[i].y));
				if (compareto.x != h_rotations[i].x || compareto.y != h_rotations[i].y || compareto.z != h_rotations[i].z)
					equalangles = false;
			}

			// Only need to rotate n times if not all angles are equal
			d_Rotate3DFT(t_targetftRe, t_targetftIm,
				d_targetftrotated,
				dimsvolume,
				h_rotations,
				equalangles ? 1 : nvolumes,
				T_INTERP_CUBIC,
				false);

			d_Rotate3DFT(t_targetpsf,
				d_buffer1,			// buffer 1 = rotated target PSF
				dimsvolume,
				h_rotations,
				equalangles ? 1 : nvolumes,
				T_INTERP_CUBIC,
				false);

			////////////////
			// Common PSF://
			////////////////

			d_ForceCommonPSF(d_volumesft, d_targetftrotated, (tcomplex*)d_buffer2, d_targetftrotated, d_volumespsfsphere, d_buffer1, d_buffer3, ElementsFFT(dimsvolume), equalangles, nvolumes);
			d_SumMonolithic(d_buffer3, d_samples, ElementsFFT(dimsvolume), nvolumes);


			////////////
			// Target://
			////////////

			// Shift if masked
			if (usevolumemask)
				d_Shift(d_targetftrotated, d_targetftrotated, dimsvolume, h_translations, false, nvolumes);
			// Transform into real-space
			d_IFFTC2R(d_targetftrotated, d_buffer1, &planback);
			if (!usevolumemask)
			{
				// Normalize
				d_NormMonolithic(d_buffer1, d_buffer1, Elements(dimsvolume), T_NORM_MEAN01STD, nvolumes);
				// Transform into Fourier-space
				d_FFTR2C(d_buffer1, d_targetftrotated, &planforw);
			}
			else
			{
				// Normalize within mask
				d_NormMonolithic(d_buffer1, (tfloat*)d_targetftrotated, Elements(dimsvolume), d_targetmask, T_NORM_MEAN01STD, nvolumes);
			}


			/////////////
			// Volumes://
			/////////////

			// Transform into real-space
			d_IFFTC2R((tcomplex*)d_buffer2, d_buffer1, &planback);
			if (!usevolumemask)
			{
				// Normalize
				d_NormMonolithic(d_buffer1, d_buffer2, Elements(dimsvolume), T_NORM_MEAN01STD, nvolumes);
				// Transform into Fourier-space
				d_FFTR2C(d_buffer2, (tcomplex*)d_buffer1, &planforw);
			}
			else
			{
				// Normalize within mask
				d_NormMonolithic(d_buffer1, d_buffer2, Elements(dimsvolume), d_targetmask, T_NORM_MEAN01STD, nvolumes);
			}


			/////////////////
			// Correlation://
			/////////////////

			if (!usevolumemask)
			{
				// Conjugate-multiply for CC
				d_ComplexMultiplyByConjVector(d_targetftrotated, (tcomplex*)d_buffer1, d_targetftrotated, ElementsFFT(dimsvolume) * nvolumes);
				// Transform CC into real-space
				d_IFFTC2R(d_targetftrotated, d_buffer1, &planback);
				// Remap zero to center
				d_RemapFullFFT2Full(d_buffer1, d_buffer2, dimsvolume, nvolumes);

				// Find peak positions and values
				d_Peak(d_buffer2, d_positions, d_scores, dimsvolume, T_PEAK_SUBCOARSE, NULL, NULL, nvolumes);
			}
			else
			{
				// Multiply for real-space CC
				d_MultiplyByVector((tfloat*)d_targetftrotated, d_buffer2, d_buffer1, Elements(dimsvolume) * nvolumes);
				// Sum values within mask
				d_SumMonolithic(d_buffer1, d_scores, d_targetmask, Elements(dimsvolume), nvolumes);
			}


			//////////////
			// Get data://
			//////////////

			tfloat3* h_positions = (tfloat3*)MallocFromDeviceArray(d_positions, nvolumes * sizeof(tfloat3));
			tfloat* h_scores = (tfloat*)MallocFromDeviceArray(d_scores, nvolumes * sizeof(tfloat));
			tfloat* h_samples = (tfloat*)MallocFromDeviceArray(d_samples, nvolumes * sizeof(tfloat));

			column_vector scores = column_vector(nvolumes);
			tfloat3 c = tfloat3(dimsvolume.x / 2, dimsvolume.y / 2, dimsvolume.z / 2);
			for (int i = 0; i < nvolumes; i++)
			{
				if (!usevolumemask)
				{
					h_results[i].translation = tfloat3(-h_positions[i].x + c.x, -h_positions[i].y + c.y, -h_positions[i].z + c.z);
					h_results[i].score = h_scores[i] / (tfloat)Elements(dimsvolume) / (tfloat)Elements(dimsvolume);	// 2nd div because of IFFTC2R without norm
				}
				else
				{
					h_results[i].translation = tfloat3((tfloat)(vec(i * 6 + 3) * 30.0), (tfloat)(vec(i * 6 + 4) * 30.0), (tfloat)(vec(i * 6 + 5) * 30.0));
					h_results[i].score = h_scores[i] / h_targetmasksum[i];
				}

				if (h_results[i].score > 1.1f)
					h_results[i].score = 0;

				h_results[i].rotation = h_rotations[i];
				h_results[i].samples = h_samples[i];
				scores(i) = 1.0 - h_results[i].score;
			}

			free(h_rotations);
			free(h_translations);
			free(h_samples);
			free(h_scores);
			free(h_positions);

			return scores;
		}

		double Evaluate(column_vector& vec)
		{
			column_vector scores = ComputeScores(vec);
			double sumscore = 0.0;
			for (int i = 0; i < nvolumes; i++)
				sumscore += scores(i);

			return sumscore / (double)nvolumes * 1.0;
		}

		virtual column_vector Derivative(column_vector x)
		{
			column_vector d = column_vector(x.size());

			int nvars = usevolumemask ? 6 : 3;
			std::vector<column_vector> v_separated;
			for (int v = 0; v < nvars; v++)
			{
				column_vector vec = column_vector(nvolumes * nvars);
				for (int i = 0; i < nvolumes; i++)
					for (int j = 0; j < nvars; j++)
						vec(i * nvars + j) = v == j ? x(i * nvars + j) + psi : x(i * nvars + j);
				v_separated.push_back(ComputeScores(vec));

				vec = column_vector(nvolumes * nvars);
				for (int i = 0; i < nvolumes; i++)
					for (int j = 0; j < nvars; j++)
						vec(i * nvars + j) = v == j ? x(i * nvars + j) - psi : x(i * nvars + j);
				v_separated.push_back(ComputeScores(vec));
			}

			column_vector neutral = ComputeScores(x);

			for (int n = 0; n < nvolumes; n++)
			{
				for (int v = 0; v < nvars; v++)
				{
					double sump = 0.0, sumn = 0.0;

					for (int i = 0; i < nvolumes; i++)
						if (i == n)
						{
							sump += v_separated[v * 2](i);
							sumn += v_separated[v * 2 + 1](i);
						}
						else
						{
							sump += neutral(i);
							sumn += neutral(i);
						}

					sump /= (double)nvolumes;
					sumn /= (double)nvolumes;

					d(n*nvars + v) = (sump - sumn) / (2.0 * psi);
				}
			}

			return d;
		}
	};


	/////////////////////////////////////////////////
	//Aligns one or multiple 3D volumes to a target//
	/////////////////////////////////////////////////

	void d_Align3D(tfloat* d_volumes, tfloat* d_volumespsf,
		tfloat* d_target, tfloat* d_targetpsf,
		tfloat* d_volumesmask,
		int3 dimsvolume,
		int nvolumes,
		tfloat angularspacing,
		tfloat2 phirange, tfloat2 thetarange, tfloat2 psirange,
		bool optimize,
		Align3DParams* h_results)
	{
		tcomplex* d_volumesft;
		cudaMalloc((void**)&d_volumesft, ElementsFFT(dimsvolume) * nvolumes * sizeof(tcomplex));
		tcomplex* d_targetft;
		cudaMalloc((void**)&d_targetft, ElementsFFT(dimsvolume) * sizeof(tcomplex));

		d_FFTR2C(d_volumes, d_volumesft, 3, dimsvolume, nvolumes);
		d_FFTR2C(d_target, d_targetft, 3, dimsvolume);
		d_RemapHalfFFT2Half(d_targetft, d_targetft, dimsvolume);
		/*CudaWriteToBinaryFile("d_volumesft.bin", d_volumesft, ElementsFFT(dimsvolume) * nvolumes * sizeof(tcomplex));
		CudaWriteToBinaryFile("d_targetft.bin", d_targetft, ElementsFFT(dimsvolume) * nvolumes * sizeof(tcomplex));*/

		Align3DOptimizer optimizer = Align3DOptimizer(d_volumesft, d_volumespsf,
			d_targetft, d_targetpsf,
			d_volumesmask,
			dimsvolume,
			nvolumes,
			thetarange);

		int numangles = 0;
		tfloat3* h_angles = GetEqualAngularSpacing(phirange, thetarange, psirange, angularspacing, numangles);
		std::vector<std::vector<Align3DParams> > v_params;
		for (int i = 0; i < nvolumes; i++)
			v_params.push_back(std::vector<Align3DParams>());

		for (int a = 0; a < numangles; a++)
		{
			column_vector angles = column_vector(nvolumes * 3);
			for (int i = 0; i < nvolumes; i++)
			{
				angles(i * 3) = h_angles[a].x;
				angles(i * 3 + 1) = h_angles[a].y;
				angles(i * 3 + 2) = h_angles[a].z;
			}

			optimizer.Evaluate(angles);		// perform CC for all volumes

			for (int i = 0; i < nvolumes; i++)
				v_params[i].push_back(optimizer.h_results[i]);

			std::cout << a << "\n";
		}
		free(h_angles);

		// sort params for each volume by their CC, descending order
		for (int i = 0; i < nvolumes; i++)
			sort(v_params[i].begin(), v_params[i].end(),
			[](const Align3DParams &a, const Align3DParams &b) -> bool
		{
			return a.score > b.score;
		});

		for (int i = 0; i < nvolumes; i++)	// after grid search, store best params for each volume as the result
			h_results[i] = v_params[i][0];

		if (optimize)
		{
			bool usevolumemask = d_volumesmask != NULL;
			int nvariables = usevolumemask ? 6 : 3;
			optimizer.usevolumemask = usevolumemask;
			for (int i = 0; i < 3; i++)	//take the 3 best orientations and optimize them
			{
				column_vector vec = column_vector(nvolumes * nvariables);
				for (int v = 0; v < nvolumes; v++)
				{
					vec(v * nvariables) = v_params[v][i].rotation.x;
					vec(v * nvariables + 1) = v_params[v][i].rotation.y;
					vec(v * nvariables + 2) = v_params[v][i].rotation.z;
					if (usevolumemask)
					{
						vec(v * nvariables + 3) = v_params[v][i].translation.x / 30.0;
						vec(v * nvariables + 4) = v_params[v][i].translation.y / 30.0;
						vec(v * nvariables + 5) = v_params[v][i].translation.z / 30.0;
					}
				}

				optimizer.psi = angularspacing / 10.0;
				dlib::find_min(dlib::bfgs_search_strategy(),
					dlib::objective_delta_stop_strategy(1e-3),
					EvalWrapper(&optimizer),
					DerivativeWrapper(&optimizer),
					vec,
					-1);

				for (int v = 0; v < nvolumes; v++)
					vec(v * nvariables + 1) = min(thetarange.y, max(thetarange.x, vec(v * nvariables + 1)));
				optimizer.Evaluate(vec);

				for (int v = 0; v < nvolumes; v++)
					if (optimizer.h_results[v].score > h_results[v].score)
						h_results[v] = optimizer.h_results[v];
			}
		}

		// Convert results from rot->trans order to trans->rot, and reverse to get volume->target
		for (int i = 0; i < nvolumes; i++)
		{
			glm::vec3 v_trans = glm::inverse(Matrix3Euler(h_results[i].rotation)) * glm::vec3(h_results[i].translation.x, h_results[i].translation.y, h_results[i].translation.z);
			v_trans = -v_trans;

			h_results[i].translation = tfloat3(v_trans.x, v_trans.y, v_trans.z);
			h_results[i].rotation = EulerInverse(h_results[i].rotation);
		}

		cudaFree(d_volumesft);
		cudaFree(d_targetft);
		optimizer.~Align3DOptimizer();
	}
}