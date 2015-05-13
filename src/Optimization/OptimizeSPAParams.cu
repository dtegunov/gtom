#include "Prerequisites.cuh"
#include "CTF.cuh"
#include "CubicInterp.cuh"
#include "FFT.cuh"
#include "Generics.cuh"
#include "Helper.cuh"
#include "ImageManipulation.cuh"
#include "Masking.cuh"
#include "Optimization.cuh"
#include "Projection.cuh"
#include "Reconstruction.cuh"

class SPAParamsOptimizer : public Optimizer
{
private:
	int nimages;
	int3 dimsimage;
	int3 dimsvolume;
	tfloat2 thetarange;
	tfloat maskradius;
	tfloat masksum;

	tcomplex* d_imagesft, *d_imagesftcentered;
	tfloat* d_imagespsf, *d_imagespsfdecentered;

	tcomplex* d_projft;
	tfloat* d_projpsf;

	tcomplex* d_compareft;
	tfloat* d_compare;

	tfloat* d_minpsf;

	tfloat* d_mask;
	tfloat* d_corrsum, *h_corrsum;

	cufftHandle planback;

public:
	SPAParamsOptimizer(tcomplex* _d_imagesft, tfloat* _d_imagespsf,
					   int3 _dimsimage,
					   int _nimages,
					   int _maskradius,
					   tfloat2 _thetarange)
	{
		d_imagesft = _d_imagesft;
		d_imagespsf = _d_imagespsf;

		dimsimage = _dimsimage;
		dimsvolume = toInt3(dimsimage.x, dimsimage.x, dimsimage.x);
		thetarange = _thetarange;
		nimages = _nimages;
		maskradius = _maskradius;
		
		cudaMalloc((void**)&d_imagesftcentered, ElementsFFT(dimsimage) * nimages * sizeof(tcomplex));
		d_RemapHalfFFT2Half(d_imagesft, d_imagesftcentered, dimsimage, nimages);
		cudaMalloc((void**)&d_imagespsfdecentered, ElementsFFT(dimsimage) * nimages * sizeof(tfloat));
		d_RemapHalf2HalfFFT(d_imagespsf, d_imagespsfdecentered, dimsimage, nimages);

		cudaMalloc((void**)&d_projft, ElementsFFT(dimsimage) * nimages * sizeof(tcomplex));
		cudaMalloc((void**)&d_projpsf, ElementsFFT(dimsimage) * nimages * sizeof(tfloat));
		cudaMalloc((void**)&d_compareft, ElementsFFT(dimsimage) * 2 * nimages * sizeof(tcomplex));
		cudaMalloc((void**)&d_compare, Elements(dimsimage) * 2 * nimages * sizeof(tfloat));
		cudaMalloc((void**)&d_minpsf, ElementsFFT(dimsimage) * nimages * sizeof(tfloat));

		// Create circular mask for comparison are in real space, calc number of samples
		{
			d_mask = CudaMallocValueFilled(Elements(dimsimage) * 2 * nimages, (tfloat)1);
			d_SphereMask(d_mask, d_mask, dimsimage, &maskradius, 0, NULL, nimages * 2);
			d_RemapFull2FullFFT(d_mask, d_mask, dimsimage, nimages * 2);
			tfloat* d_masksum = CudaMallocValueFilled(1, (tfloat)0);
			d_Sum(d_mask, d_masksum, Elements(dimsimage), 1);
			cudaMemcpy(&masksum, d_masksum, sizeof(tfloat), cudaMemcpyDeviceToHost);
			cudaFree(d_masksum);
		}

		cudaMalloc((void**)&d_corrsum, nimages * sizeof(tfloat));
		h_corrsum = (tfloat*)malloc(nimages * sizeof(tfloat));

		planback = d_IFFTC2RGetPlan(2, dimsimage, nimages * 2);
	}

	~SPAParamsOptimizer()
	{
		cufftDestroy(planback);

		cudaFree(d_corrsum);
		cudaFree(d_mask);
		cudaFree(d_minpsf);
		cudaFree(d_compareft);
		cudaFree(d_compare);
		cudaFree(d_projpsf);
		cudaFree(d_projft);
		cudaFree(d_imagespsfdecentered);
		cudaFree(d_imagesftcentered);

		free(h_corrsum);
	}

	double ComputeScore(tcomplex* d_volumeft, tfloat* d_volumepsf, tfloat3* h_angles, tfloat2* h_shifts)
	{
		// Project reconstructed volume, equalize PSF with originals, transform into real space
		d_ProjForward(d_volumeft, d_volumepsf, dimsvolume, d_projft, d_projpsf, h_angles, h_shifts, T_INTERP_LINEAR, false, nimages);
		d_ForceCommonPSF(d_imagesft, d_projft, d_compareft, d_compareft + ElementsFFT(dimsimage) * nimages, d_imagespsfdecentered, d_projpsf, d_minpsf, ElementsFFT(dimsimage), false, nimages);
		d_IFFTC2R(d_compareft, d_compare, &planback);
		/*CudaWriteToBinaryFile("d_compare.bin", d_compare, Elements(dimsimage) * nimages * 2 * sizeof(tfloat));
		CudaWriteToBinaryFile("d_minpsf.bin", d_minpsf, ElementsFFT(dimsimage) * nimages * sizeof(tfloat));
		CudaWriteToBinaryFile("d_imagespsf.bin", d_imagespsfdecentered, ElementsFFT(dimsimage) * nimages * sizeof(tfloat));
		CudaWriteToBinaryFile("d_projpsf.bin", d_projpsf, ElementsFFT(dimsimage) * nimages * sizeof(tfloat));
		CudaWriteToBinaryFile("d_volumepsf.bin", d_volumepsf, ElementsFFT(dimsvolume) * sizeof(tfloat));
		CudaWriteToBinaryFile("d_volumeft.bin", d_volumeft, ElementsFFT(dimsvolume) * sizeof(tcomplex));*/

		// Real-space CC within the masked area
		d_NormMonolithic(d_compare, d_compare, Elements(dimsimage), d_mask, T_NORM_MEAN01STD, nimages * 2);
		d_MultiplyByVector(d_compare, d_compare + Elements(dimsimage) * nimages, d_compare, Elements(dimsimage) * nimages, 1);
		//CudaWriteToBinaryFile("d_compare.bin", d_compare, Elements(dimsimage) * nimages * 2 * sizeof(tfloat));
		d_SumMonolithic(d_compare, d_corrsum, d_mask, Elements(dimsimage), nimages);
		cudaMemcpy(h_corrsum, d_corrsum, nimages * sizeof(tfloat), cudaMemcpyDeviceToHost);

		// Sum up and normalize CC scores
		double sum = 0;
		for (int n = 0; n < nimages; n++)
			sum += h_corrsum[n];
		sum /= masksum * (double)nimages;
		sum = 1.0 - sum;

		return sum * 10.0;
	}

	double Evaluate(column_vector& vec)
	{
		/*for (int n = 0; n < nimages; n++)
			vec(n * 5 + 1) = min(thetarange.y, max(vec(n * 5 + 1), thetarange.x));*/

		tfloat3* h_angles = (tfloat3*)malloc(nimages * sizeof(tfloat3));
		tfloat2* h_shifts = (tfloat2*)malloc(nimages * sizeof(tfloat2));
		for (int n = 0; n < nimages; n++)
		{
			h_angles[n] = tfloat3((tfloat)vec(n * 5), (tfloat)vec(n * 5 + 1), (tfloat)vec(n * 5 + 2));
			h_shifts[n] = tfloat2((tfloat)vec(n * 5 + 3), (tfloat)vec(n * 5 + 4));
		}

		tcomplex* d_volumeft = CudaMallocValueFilled(ElementsFFT(dimsvolume), make_cuComplex(0, 0));
		tfloat* d_volumepsf = CudaMallocValueFilled(ElementsFFT(dimsvolume), (tfloat)0);

		d_ReconstructFourierSincAdd(d_volumeft, d_volumepsf, dimsvolume, d_imagesftcentered, d_imagespsf, h_angles, h_shifts, nimages, true, true);
		double score = this->ComputeScore(d_volumeft, d_volumepsf, h_angles, h_shifts);

		cudaFree(d_volumepsf);
		cudaFree(d_volumeft);
		free(h_shifts);
		free(h_angles);

		return score;
	}
};

class SPAParamsSIRTOptimizer : public Optimizer
{
private:
	int nimages;
	int2 dimsimage;
	int3 dimsvolume;
	tfloat maskradius;
	tfloat masksum;

	column_vector constraintlower;
	column_vector constraintupper;

	tfloat* d_images;

	tfloat* d_volume, *d_volumeresidue;

	tfloat* d_mask;
	tfloat* d_corrsum, *h_corrsum;

	cufftHandle planback;

public:
	SPAParamsSIRTOptimizer(tfloat* _d_images,
		int2 _dimsimage,
		int _nimages,
		column_vector _constraintlower,
		column_vector _constraintupper,
		int _maskradius)
	{
		dimsimage = _dimsimage;
		dimsvolume = toInt3(dimsimage.x, dimsimage.x, dimsimage.x);
		nimages = _nimages;
		maskradius = _maskradius;

		constraintlower = _constraintlower;
		constraintupper = _constraintupper;

		d_images = _d_images;

		cudaMalloc((void**)&d_volume, Elements(dimsvolume) * sizeof(tfloat));
		cudaMalloc((void**)&d_volumeresidue, Elements(dimsvolume) * sizeof(tfloat));

		// Create circular mask for comparison in real space, calc number of samples
		{
			d_mask = CudaMallocValueFilled(Elements(dimsvolume), (tfloat)1);
			d_SphereMask(d_mask, d_mask, dimsvolume, &maskradius, 0, NULL);
			tfloat* d_masksum = CudaMallocValueFilled(1, (tfloat)0);
			d_Sum(d_mask, d_masksum, Elements(dimsvolume), 1);
			cudaMemcpy(&masksum, d_masksum, sizeof(tfloat), cudaMemcpyDeviceToHost);
			cudaFree(d_masksum);
		}

		cudaMalloc((void**)&d_corrsum, sizeof(tfloat));
		h_corrsum = (tfloat*)malloc(sizeof(tfloat));
	}

	~SPAParamsSIRTOptimizer()
	{
		cufftDestroy(planback);

		cudaFree(d_volumeresidue);
		cudaFree(d_volume);
		cudaFree(d_corrsum);
		cudaFree(d_mask);

		free(h_corrsum);
	}
	
	double Evaluate(column_vector& vec)
	{
		tfloat3* h_angles = (tfloat3*)malloc(nimages * sizeof(tfloat3));
		tfloat2* h_shifts = (tfloat2*)malloc(nimages * sizeof(tfloat2));
		tfloat2* h_intensities = (tfloat2*)malloc(nimages * sizeof(tfloat2));
		for (int n = 0; n < nimages; n++)
		{
			h_angles[n] = tfloat3((tfloat)vec(n * 7), (tfloat)vec(n * 7 + 1), (tfloat)vec(n * 7 + 2));
			h_shifts[n] = tfloat2((tfloat)vec(n * 7 + 3), (tfloat)vec(n * 7 + 4));
			h_intensities[n] = tfloat2((tfloat)vec(n * 7 + 5), (tfloat)vec(n * 7 + 6));
		}
		tfloat2* h_scales = (tfloat2*)MallocValueFilled(nimages * 2, (tfloat)1);

		d_ValueFill(d_volume, Elements(dimsvolume), (tfloat)0);
		d_RecSIRT(d_volume, d_volumeresidue, dimsvolume, tfloat3(0, 0, 0), d_images, dimsimage, nimages, h_angles, h_shifts, h_scales, h_intensities, T_INTERP_CUBIC, 1, 300, true);
		//CudaWriteToBinaryFile("d_volume.bin", d_volume, Elements(dimsvolume) * sizeof(tfloat));
		d_SumMonolithic(d_volumeresidue, d_corrsum, d_mask, Elements(dimsvolume), 1);
		cudaMemcpy(h_corrsum, d_corrsum, sizeof(tfloat), cudaMemcpyDeviceToHost);

		free(h_scales);
		free(h_intensities);
		free(h_shifts);
		free(h_angles);

		return h_corrsum[0] / masksum * 100.0;
	}

	column_vector Derivative(column_vector x)
	{
		column_vector d = column_vector(x.size());
		for (int i = 0; i < x.size(); i++)
			d(i) = 0.0;

		for (int i = 0; i < x.size(); i++)
		{
			if (abs(constraintupper(i) - constraintlower(i)) < 1e-8)
				continue;

			double old = x(i);

			x(i) = old + psi;
			double dp = this->Evaluate(x);
			x(i) = old - psi;
			double dn = this->Evaluate(x);

			x(i) = old;
			d(i) = (dp - dn) / (2.0 * psi);
		}

		return d;
	}
};

//void d_OptimizeSPAParams(tfloat* d_images, tfloat* d_imagespsf, int2 dimsimage, int nimages, int maskradius, tfloat3* h_angles, tfloat2* h_shifts, tfloat2 thetarange, tfloat &finalscore)
//{
//	d_RemapFull2FullFFT(d_images, d_images, toInt3(dimsimage), nimages);
//
//	tcomplex* d_imagesft;
//	cudaMalloc((void**)&d_imagesft, ElementsFFT2(dimsimage) * nimages * sizeof(tcomplex));
//	d_FFTR2C(d_images, d_imagesft, 2, toInt3(dimsimage), nimages);
//
//	column_vector vec = column_vector(nimages * 5);
//	for (int n = 0; n < nimages; n++)
//	{
//		vec(n * 5) = h_angles[n].x;
//		vec(n * 5 + 1) = h_angles[n].y;
//		vec(n * 5 + 2) = h_angles[n].z;
//		vec(n * 5 + 3) = h_shifts[n].x;
//		vec(n * 5 + 4) = h_shifts[n].y;
//	}
//
//	column_vector constraintlow = column_vector(nimages * 5);
//	column_vector constraintupper = column_vector(nimages * 5);
//	for (int n = 0; n < nimages; n++)
//	{
//		constraintlow(n * 5) = -PI2;
//		constraintlow(n * 5 + 1) = thetarange.x;
//		constraintlow(n * 5 + 2) = -PI2;
//		constraintlow(n * 5 + 3) = -4.0;
//		constraintlow(n * 5 + 4) = -4.0;
//
//		constraintupper(n * 5) = PI2;
//		constraintupper(n * 5 + 1) = thetarange.y;
//		constraintupper(n * 5 + 2) = PI2;
//		constraintupper(n * 5 + 3) = 4.0;
//		constraintupper(n * 5 + 4) = 4.0;
//	}
//
//	SPAParamsOptimizer optimizer = SPAParamsOptimizer(d_imagesft, d_imagespsf, toInt3(dimsimage), nimages, maskradius, thetarange);
//
//	double initialscore = optimizer.Evaluate(vec);
//
//	optimizer.psi = ToRad(2);
//	dlib::find_min_box_constrained(dlib::bfgs_search_strategy(),
//								   dlib::objective_delta_stop_strategy(1e-8),
//								   EvalWrapper(&optimizer),
//								   DerivativeWrapper(&optimizer),
//								   vec,
//								   constraintlow, constraintupper);
//
//	// Store results
//	finalscore = optimizer.Evaluate(vec);
//	for (int n = 0; n < nimages; n++)
//	{
//		h_angles[n] = tfloat3(ToDeg((tfloat)vec(n * 5)), ToDeg((tfloat)vec(n * 5 + 1)), ToDeg((tfloat)vec(n * 5 + 2)));
//		h_shifts[n] = tfloat2((tfloat)vec(n * 5 + 3), (tfloat)vec(n * 5 + 4));
//	}
//
//	cudaFree(d_imagesft);
//	d_RemapFull2FullFFT(d_images, d_images, toInt3(dimsimage), nimages);
//}

void d_OptimizeSPAParams(tfloat* d_images, 
						 tfloat* d_imagespsf, 
						 int2 dimsimage, 
						 int nimages, 
						 int indexhold,
						 int maskradius, 
						 tfloat3* h_angles, 
						 tfloat2* h_shifts, 
						 tfloat2* h_intensities, 
						 tfloat3 deltaanglesmax,
						 tfloat deltashiftmax,
						 tfloat2 deltaintensitymax, 
						 tfloat &finalscore)
{
	column_vector vec = column_vector(nimages * 7);
	for (int n = 0; n < nimages; n++)
	{
		vec(n * 7) = h_angles[n].x;
		vec(n * 7 + 1) = h_angles[n].y;
		vec(n * 7 + 2) = h_angles[n].z;
		vec(n * 7 + 3) = h_shifts[n].x;
		vec(n * 7 + 4) = h_shifts[n].y;
		vec(n * 7 + 5) = h_intensities[n].x;
		vec(n * 7 + 6) = h_intensities[n].y;
	}

	column_vector constraintlower = column_vector(nimages * 7);
	column_vector constraintupper = column_vector(nimages * 7);
	for (int n = 0; n < nimages; n++)
	{
		constraintlower(n * 7) = h_angles[n].x - (n == indexhold ? 0.0 : deltaanglesmax.x);
		constraintlower(n * 7 + 1) = h_angles[n].y - (n == indexhold ? 0.0 : deltaanglesmax.y);
		constraintlower(n * 7 + 2) = h_angles[n].z - (n == indexhold ? 0.0 : deltaanglesmax.z);
		constraintlower(n * 7 + 3) = h_shifts[n].x - (n == indexhold ? 0.0 : deltashiftmax);
		constraintlower(n * 7 + 4) = h_shifts[n].y - (n == indexhold ? 0.0 : deltashiftmax);
		constraintlower(n * 7 + 5) = h_intensities[n].x - (n == indexhold ? 0.0 : deltaintensitymax.x);
		constraintlower(n * 7 + 6) = h_intensities[n].y - (n == indexhold ? 0.0 : deltaintensitymax.y);

		constraintupper(n * 7) = h_angles[n].x + (n == indexhold ? 0.0 : deltaanglesmax.x);
		constraintupper(n * 7 + 1) = h_angles[n].y + (n == indexhold ? 0.0 : deltaanglesmax.y);
		constraintupper(n * 7 + 2) = h_angles[n].z + (n == indexhold ? 0.0 : deltaanglesmax.z);
		constraintupper(n * 7 + 3) = h_shifts[n].x + (n == indexhold ? 0.0 : deltashiftmax);
		constraintupper(n * 7 + 4) = h_shifts[n].y + (n == indexhold ? 0.0 : deltashiftmax);
		constraintupper(n * 7 + 5) = h_intensities[n].x + (n == indexhold ? 0.0 : deltaintensitymax.x);
		constraintupper(n * 7 + 6) = h_intensities[n].y + (n == indexhold ? 0.0 : deltaintensitymax.y);
	}

	SPAParamsSIRTOptimizer optimizer(d_images, dimsimage, nimages, constraintlower, constraintupper, maskradius);

	double initialscore = optimizer.Evaluate(vec);

	optimizer.psi = ToRad(0.8);
	dlib::find_min_box_constrained(dlib::bfgs_search_strategy(),
		dlib::objective_delta_stop_strategy(1e-5),
		EvalWrapper(&optimizer),
		DerivativeWrapper(&optimizer),
		vec,
		constraintlower, constraintupper);

	// Store results
	finalscore = optimizer.Evaluate(vec);
	for (int n = 0; n < nimages; n++)
	{
		h_angles[n] = tfloat3(ToDeg((tfloat)vec(n * 5)), ToDeg((tfloat)vec(n * 5 + 1)), ToDeg((tfloat)vec(n * 5 + 2)));
		h_shifts[n] = tfloat2((tfloat)vec(n * 5 + 3), (tfloat)vec(n * 5 + 4));
	}
}