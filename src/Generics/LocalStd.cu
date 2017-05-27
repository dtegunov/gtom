#include "Prerequisites.cuh"
#include "FFT.cuh"
#include "Generics.cuh"
#include "Helper.cuh"
#include "Masking.cuh"


namespace gtom
{
	void d_LocalStd(tfloat* d_map, int3 dimsmap, tfloat localradius, tfloat* d_std, tfloat* d_mean, cufftHandle planforw, cufftHandle planback)
	{
		cufftHandle localplanforw = planforw;
		if (planforw == NULL)
			localplanforw = d_FFTR2CGetPlan(DimensionCount(dimsmap), dimsmap, 1);
		cufftHandle localplanback = planback;
		if (planback == NULL)
			localplanback = d_IFFTC2RGetPlan(DimensionCount(dimsmap), dimsmap, 1);

		tcomplex* d_maskft = CudaMallocValueFilled(ElementsFFT(dimsmap), make_cuComplex(1, 1));
		tfloat masksum = 0;

		// Create spherical mask, calculate its sum, and pre-FFT it for convolution
		{
			d_SphereMask((tfloat*)d_maskft, (tfloat*)d_maskft, dimsmap, &localradius, 0, NULL);
			d_RemapFull2FullFFT((tfloat*)d_maskft, (tfloat*)d_maskft, dimsmap);

			tfloat* d_sum = CudaMallocValueFilled(1, (tfloat)0);
			d_Sum((tfloat*)d_maskft, d_sum, Elements(dimsmap));
			cudaMemcpy(&masksum, d_sum, sizeof(tfloat), cudaMemcpyDeviceToHost);
			cudaFree(d_sum);

			d_FFTR2C((tfloat*)d_maskft, d_maskft, &localplanforw);
		}

		tcomplex* d_mapft;
		cudaMalloc((void**)&d_mapft, ElementsFFT(dimsmap) * sizeof(tcomplex));
		tcomplex* d_map2ft;
		cudaMalloc((void**)&d_map2ft, ElementsFFT(dimsmap) * sizeof(tcomplex));
		
		// Create FTs of map and map^2
		{
			d_FFTR2C(d_map, d_mapft, &localplanforw);

			d_Square(d_map, (tfloat*)d_map2ft, Elements(dimsmap));
			d_FFTR2C((tfloat*)d_map2ft, d_map2ft, &localplanforw);
		}

		// Convolute
		{
			d_ComplexMultiplyByConjVector(d_mapft, d_maskft, d_mapft, ElementsFFT(dimsmap));
			d_ComplexMultiplyByConjVector(d_map2ft, d_maskft, d_map2ft, ElementsFFT(dimsmap));

			d_IFFTC2R(d_mapft, (tfloat*)d_mapft, &localplanback, dimsmap);
			d_IFFTC2R(d_map2ft, (tfloat*)d_map2ft, &localplanback, dimsmap);
		}

		// Optionally, also output local mean
		if (d_mean != NULL)
		{
			d_DivideByScalar((tfloat*)d_mapft, d_mean, Elements(dimsmap), masksum);
		}

		// std = sqrt(max(0, masksum * conv2 - conv1^2)) / masksum
		{
			d_MultiplyByScalar((tfloat*)d_map2ft, (tfloat*)d_map2ft, Elements(dimsmap), masksum);
			d_Square((tfloat*)d_mapft, (tfloat*)d_mapft, Elements(dimsmap));

			d_SubtractVector((tfloat*)d_map2ft, (tfloat*)d_mapft, (tfloat*)d_map2ft, Elements(dimsmap));
			d_MaxOp((tfloat*)d_map2ft, (tfloat)0, (tfloat*)d_map2ft, Elements(dimsmap));

			d_Sqrt((tfloat*)d_map2ft, (tfloat*)d_map2ft, Elements(dimsmap));

			d_DivideByScalar((tfloat*)d_map2ft, d_std, Elements(dimsmap), masksum);
		}

		cudaFree(d_map2ft);
		cudaFree(d_mapft);
		cudaFree(d_maskft);

		if (planforw == NULL)
			cufftDestroy(localplanforw);
		if (planback == NULL)
			cufftDestroy(localplanback);
	}
}