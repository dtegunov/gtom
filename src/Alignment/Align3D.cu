#include "Prerequisites.cuh"
#include "Alignment.cuh"
#include "Angles.cuh"
#include "Correlation.cuh"
#include "CubicInterp.cuh"
#include "FFT.cuh"
#include "Generics.cuh"
#include "Helper.cuh"
#include "ImageManipulation.cuh"
#include "Masking.cuh"
#include "Optimization.cuh"
#include "Transformation.cuh"

__global__ void Align3DCompositeKernel(tfloat* d_volumespsf, tfloat* d_targetpsf, tfloat* d_minpsf, tcomplex* d_involumesft, tcomplex* d_outvolumesft, tcomplex* d_intargetft, tcomplex* d_outtargetft, uint n);

class Align3DOptimizer : public Optimizer
{
private:
	int nvolumes;
	int3 dimsvolume;

	tcomplex* d_volumesftrotated;
	tfloat* d_targetpsfsphere;
	tfloat* d_buffer1;
	tfloat* d_buffer2;
	tfloat* d_samples;

	cudaTextureObject_t* t_volumesftRe, *t_volumesftIm, *t_volumespsf;
	tcomplex* d_targetft;
	tfloat* d_targetpsf;
	tfloat* d_mask;

	tfloat* d_scores;
	tfloat3* d_positions;

	tfloat* d_targetmask;
	tfloat* h_targetmasksum;

	cufftHandle planvolumesforw, planvolumesback;
	cufftHandle plantargetforw, plantargetback;

public:
	Align3DParams* h_results;
	bool usevolumemask;

	Align3DOptimizer(cudaTextureObject_t* _t_volumesftRe, cudaTextureObject_t* _t_volumesftIm, cudaTextureObject_t* _t_volumespsf,
					 tcomplex* _d_targetft, tfloat* _d_targetpsf,
					 tfloat* _d_targetmask,
					 int3 _dimsvolume,
					 int _nvolumes)
	{
		nvolumes = _nvolumes;
		dimsvolume = _dimsvolume;
		t_volumesftRe = _t_volumesftRe;
		t_volumesftIm = _t_volumesftIm;
		t_volumespsf = _t_volumespsf;
		d_targetft = _d_targetft;
		d_targetpsf = _d_targetpsf;

		cudaMalloc((void**)&d_volumesftrotated, ElementsFFT(dimsvolume) * nvolumes * sizeof(tcomplex));
		cudaMalloc((void**)&d_targetpsfsphere, ElementsFFT(dimsvolume) * sizeof(tfloat));
		cudaMalloc((void**)&d_buffer1, ElementsFFT(dimsvolume) * nvolumes * sizeof(tcomplex));
		cudaMalloc((void**)&d_buffer2, ElementsFFT(dimsvolume) * nvolumes * sizeof(tcomplex));
		cudaMalloc((void**)&d_samples, nvolumes * sizeof(tfloat));

		cudaMalloc((void**)&d_positions, nvolumes * sizeof(tfloat3));
		cudaMalloc((void**)&d_scores, nvolumes * sizeof(tfloat));

		h_results = (Align3DParams*)malloc(nvolumes * sizeof(Align3DParams));
		usevolumemask = false;

		d_mask = CudaMallocValueFilled(Elements(dimsvolume), (tfloat)1);
		d_SphereMask(d_mask, d_mask, dimsvolume, NULL, 0, NULL);
		d_RemapFull2HalfFFT(d_mask, d_mask, dimsvolume);
		d_MultiplyByVector(d_targetpsf, d_mask, d_targetpsfsphere, ElementsFFT(dimsvolume));

		d_targetmask = _d_targetmask;
		if (d_targetmask != NULL)
		{
			tfloat* d_result = CudaMallocValueFilled(nvolumes, (tfloat)0);
			d_Sum(d_targetmask, d_result, Elements(dimsvolume), nvolumes);
			cudaMemcpy(h_targetmasksum, d_result, nvolumes * sizeof(tfloat), cudaMemcpyDeviceToHost);
			cudaFree(d_result);
		}

		planvolumesforw = d_FFTR2CGetPlan(3, dimsvolume, nvolumes);
		planvolumesback = d_IFFTC2RGetPlan(3, dimsvolume, nvolumes);
		plantargetforw = d_FFTR2CGetPlan(3, dimsvolume, 1);
		plantargetback = d_IFFTC2RGetPlan(3, dimsvolume, 1);
	}

	~Align3DOptimizer()
	{
		free(h_results);

		cudaFree(d_mask);
		cudaFree(d_samples);
		cudaFree(d_scores);
		cudaFree(d_positions);
		cudaFree(d_buffer2);
		cudaFree(d_buffer1);
		cudaFree(d_targetpsfsphere);
		cudaFree(d_volumesftrotated);

		cufftDestroy(planvolumesforw);
		cufftDestroy(planvolumesback);
		cufftDestroy(plantargetforw);
		cufftDestroy(plantargetback);
	}

	column_vector ComputeScores(column_vector& vec)
	{
		////////////////////
		// Rotate volumes://
		////////////////////

		tfloat3* h_translations = (tfloat3*)malloc(nvolumes * sizeof(tfloat3));
		for (int i = 0; i < nvolumes; i++)
		{
			tfloat3 rotation;
			if (!usevolumemask)
				rotation = tfloat3((tfloat)vec(i * 3), (tfloat)vec(i * 3 + 1), (tfloat)vec(i * 3 + 2));
			else
			{
				rotation = tfloat3((tfloat)vec(i * 6), (tfloat)vec(i * 6 + 1), (tfloat)vec(i * 6 + 2));
				h_translations[i] = tfloat3((tfloat)(vec(i * 6 + 3) * 30.0), (tfloat)(vec(i * 6 + 4) * 30.0), (tfloat)(vec(i * 6 + 5) * 30.0));
			}

			d_Rotate3DFT(t_volumesftRe[i], t_volumesftIm[i],
				d_volumesftrotated + ElementsFFT(dimsvolume) * i,
				dimsvolume,
				&rotation,
				1,
				T_INTERP_CUBIC,
				false);

			d_Rotate3DFT(t_volumespsf[i],
				d_buffer1 + ElementsFFT(dimsvolume) * i,			// buffer 1 = rotated volume PSF
				dimsvolume,
				&rotation,
				1,
				T_INTERP_CUBIC,
				false);
		}


		////////////////
		// Common PSF://
		////////////////

		dim3 TpB = min(192, NextMultipleOf(ElementsFFT(dimsvolume), 32));
		dim3 grid = dim3(min((ElementsFFT(dimsvolume) + TpB.x - 1) / TpB.x, 8), nvolumes);
		Align3DCompositeKernel <<<grid, TpB>>> (d_buffer1, d_targetpsfsphere, d_buffer1, d_volumesftrotated, d_volumesftrotated, d_targetft, (tcomplex*)d_buffer2, ElementsFFT(dimsvolume));
		d_SumMonolithic(d_buffer1, d_samples, ElementsFFT(dimsvolume), nvolumes);


		/////////////
		// Volumes://
		/////////////

		// Shift if masked
		if (usevolumemask)
			d_Shift(d_volumesftrotated, d_volumesftrotated, dimsvolume, h_translations, false, nvolumes);
		// Transform into real-space
		d_IFFTC2R(d_volumesftrotated, d_buffer1, &planvolumesback);
		if (!usevolumemask)
		{
			// Normalize
			d_NormMonolithic(d_buffer1, d_buffer1, Elements(dimsvolume), T_NORM_MEAN01STD, nvolumes);
			// Transform into Fourier-space
			d_FFTR2C(d_buffer1, d_volumesftrotated, &planvolumesforw);
		}
		else
		{
			// Normalize within mask
			d_NormMonolithic(d_buffer1, (tfloat*)d_volumesftrotated, Elements(dimsvolume), d_targetmask, T_NORM_MEAN01STD, nvolumes);
		}


		////////////
		// Target://
		////////////

		// Transform into real-space
		d_IFFTC2R((tcomplex*)d_buffer2, d_buffer1, &plantargetback);
		if (!usevolumemask)
		{
			// Normalize
			d_NormMonolithic(d_buffer1, d_buffer2, Elements(dimsvolume), T_NORM_MEAN01STD, 1);
			// Transform into Fourier-space
			d_FFTR2C(d_buffer2, (tcomplex*)d_buffer1, &plantargetforw);
		}
		else
		{
			// Normalize within mask
			d_NormMonolithic(d_buffer1, d_buffer2, Elements(dimsvolume), d_targetmask, T_NORM_MEAN01STD, 1);
		}


		/////////////////
		// Correlation://
		/////////////////

		if (!usevolumemask)
		{
			// Conjugate-multiply for CC
			d_ComplexMultiplyByConjVector(d_volumesftrotated, (tcomplex*)d_buffer1, d_volumesftrotated, ElementsFFT(dimsvolume), nvolumes);
			// Transform CC into real-space
			d_IFFTC2R(d_volumesftrotated, d_buffer1, &planvolumesback);
			// Remap zero to center
			d_RemapFullFFT2Full(d_buffer1, d_buffer2, dimsvolume, nvolumes);

			// Find peak positions and values
			d_Peak(d_buffer2, d_positions, d_scores, dimsvolume, T_PEAK_SUBCOARSE, NULL, NULL, nvolumes);
		}
		else
		{
			// Multiply for real-space CC
			d_MultiplyByVector((tfloat*)d_volumesftrotated, d_buffer2, d_buffer1, Elements(dimsvolume), nvolumes);
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
				h_results[i].rotation = tfloat3((tfloat)vec(i * 3), (tfloat)vec(i * 3 + 1), (tfloat)vec(i * 3 + 2));
				h_results[i].translation = tfloat3(-h_positions[i].x + c.x, -h_positions[i].y + c.y, -h_positions[i].z + c.z);
				h_results[i].score = h_scores[i] / (tfloat)Elements(dimsvolume) / (tfloat)Elements(dimsvolume);	// 2nd div because of IFFTC2R without norm
			}
			else
			{
				h_results[i].rotation = tfloat3((tfloat)vec(i * 6), (tfloat)vec(i * 6 + 1), (tfloat)vec(i * 6 + 2));
				h_results[i].translation = tfloat3((tfloat)(vec(i * 6 + 3) * 30.0), (tfloat)(vec(i * 6 + 4) * 30.0), (tfloat)(vec(i * 6 + 5) * 30.0));
				h_results[i].score = h_scores[i] / h_targetmasksum[i];
			}

			h_results[i].samples = h_samples[i];
			scores(i) = 1.0 - h_results[i].score;
		}

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
		vector<column_vector> v_separated;
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

				for (int i = 0; i < nvolumes;i++)
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

__global__ void Align3DCompositeKernel(tfloat* d_volumespsf, tfloat* d_targetpsf, tfloat* d_minpsf, tcomplex* d_involumesft, tcomplex* d_outvolumesft, tcomplex* d_intargetft, tcomplex* d_outtargetft, uint n)
{
	d_volumespsf += n * blockIdx.y;
	d_minpsf += n * blockIdx.y;
	d_involumesft += n * blockIdx.y;
	d_outvolumesft += n * blockIdx.y;

	uint blockelements = blockDim.x * gridDim.x;
	for (uint idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockelements)
	{
		tfloat psf1 = abs(d_volumespsf[idx]);
		tfloat psf2 = abs(d_targetpsf[idx]);
		tfloat minpsf = min(psf1, psf2);
		tfloat conv1 = 0, conv2 = 0;
		if (psf1 > 0)
			conv1 = minpsf / psf1;
		if (psf2 > 0)
			conv2 = minpsf / psf2;

		tcomplex ft1 = d_involumesft[idx];
		d_outvolumesft[idx] = make_cuComplex(ft1.x * conv1, ft1.y * conv1);
		tcomplex ft2 = d_intargetft[idx];
		d_outtargetft[idx] = make_cuComplex(ft2.x * conv2, ft2.y * conv2);
		d_minpsf[idx] = minpsf;
	}
}


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
	d_RemapHalf2HalfFFT(d_targetpsf, d_targetpsf, dimsvolume, 1);	// PSFs come zero-centered

	tcomplex* d_volumesft;
	cudaMalloc((void**)&d_volumesft, ElementsFFT(dimsvolume) * nvolumes * sizeof(tcomplex));
	tcomplex* d_targetft;
	cudaMalloc((void**)&d_targetft, ElementsFFT(dimsvolume) * sizeof(tcomplex));

	d_FFTR2C(d_volumes, d_volumesft, 3, dimsvolume, nvolumes);
	d_RemapHalfFFT2Half(d_volumesft, d_volumesft, dimsvolume, nvolumes);
	d_FFTR2C(d_target, d_targetft, 3, dimsvolume);
	/*CudaWriteToBinaryFile("d_volumesft.bin", d_volumesft, ElementsFFT(dimsvolume) * nvolumes * sizeof(tcomplex));
	CudaWriteToBinaryFile("d_targetft.bin", d_targetft, ElementsFFT(dimsvolume) * nvolumes * sizeof(tcomplex));*/

	// Convert volume data to 3D textures:
	cudaTextureObject_t* t_volumesftRe = (cudaTextureObject_t*)malloc(nvolumes * sizeof(cudaTextureObject_t));
	cudaTextureObject_t* t_volumesftIm = (cudaTextureObject_t*)malloc(nvolumes * sizeof(cudaTextureObject_t));
	cudaTextureObject_t* t_volumespsf = (cudaTextureObject_t*)malloc(nvolumes * sizeof(cudaTextureObject_t));
	cudaArray_t* a_volumesftRe = (cudaArray_t*)malloc(nvolumes * sizeof(cudaArray_t));
	cudaArray_t* a_volumesftIm = (cudaArray_t*)malloc(nvolumes * sizeof(cudaArray_t));
	cudaArray_t* a_volumespsf = (cudaArray_t*)malloc(nvolumes * sizeof(cudaArray_t));
	{
		tfloat* d_tempRe, *d_tempIm;
		cudaMalloc((void**)&d_tempRe, ElementsFFT(dimsvolume) * sizeof(tfloat));
		cudaMalloc((void**)&d_tempIm, ElementsFFT(dimsvolume) * sizeof(tfloat));

		for (int n = 0; n < nvolumes; n++)
		{
			int3 dimsfft = toInt3(dimsvolume.x / 2 + 1, dimsvolume.y, dimsvolume.z);

			d_ConvertTComplexToSplitComplex(d_volumesft + ElementsFFT(dimsvolume) * n, d_tempRe, d_tempIm, ElementsFFT(dimsvolume));
			d_CubicBSplinePrefilter3D(d_tempRe, dimsfft.x * sizeof(tfloat), dimsfft);
			d_CubicBSplinePrefilter3D(d_tempIm, dimsfft.x * sizeof(tfloat), dimsfft);
			d_BindTextureTo3DArray(d_tempRe, a_volumesftRe[n], t_volumesftRe[n], dimsfft, cudaFilterModeLinear, false);
			d_BindTextureTo3DArray(d_tempIm, a_volumesftIm[n], t_volumesftIm[n], dimsfft, cudaFilterModeLinear, false);

			cudaMemcpy(d_tempRe, d_volumespsf + ElementsFFT(dimsvolume) * n, ElementsFFT(dimsvolume) * sizeof(tfloat), cudaMemcpyDeviceToDevice);
			d_CubicBSplinePrefilter3D(d_tempRe, dimsfft.x * sizeof(tfloat), dimsfft);
			d_BindTextureTo3DArray(d_tempRe, a_volumespsf[n], t_volumespsf[n], dimsfft, cudaFilterModeLinear, false);
		}

		cudaFree(d_tempRe);
		cudaFree(d_tempIm);
	}
	cudaFree(d_volumesft);

	Align3DOptimizer optimizer = Align3DOptimizer(t_volumesftRe, t_volumesftIm, t_volumespsf,
												  d_targetft, d_targetpsf,
												  d_volumesmask,
												  dimsvolume,
												  nvolumes);

	int numangles = 0;
	tfloat3* h_angles = GetEqualAngularSpacing(phirange, thetarange, psirange, angularspacing, numangles);
	vector<vector<Align3DParams>> v_params;
	for (int i = 0; i < nvolumes; i++)
		v_params.push_back(vector<Align3DParams>());

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
	}
	free(h_angles);

	// sort params for each volume by their CC, descending order
	for (int i = 0; i < nvolumes;i++)
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
						   dlib::objective_delta_stop_strategy(1e-4),
						   EvalWrapper(&optimizer),
						   DerivativeWrapper(&optimizer),
						   vec,
						   -1);

			for (int v = 0; v < nvolumes; v++)
				if (optimizer.h_results[v].score > h_results[v].score)
					h_results[v] = optimizer.h_results[v];
		}
	}

	for (int n = 0; n < nvolumes; n++)
	{
		cudaDestroyTextureObject(t_volumesftRe[n]);
		cudaDestroyTextureObject(t_volumesftIm[n]);
		cudaDestroyTextureObject(t_volumespsf[n]);
		cudaFreeArray(a_volumesftRe[n]);
		cudaFreeArray(a_volumesftIm[n]);
		cudaFreeArray(a_volumespsf[n]);
	}
	free(t_volumesftRe);
	free(t_volumesftIm);
	free(t_volumespsf);
	free(a_volumesftRe);
	free(a_volumesftIm);
	free(a_volumespsf);

	cudaFree(d_targetft);
}