#include "Prerequisites.cuh"
#include "Angles.cuh"
#include "Binary.cuh"
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
#include "Transformation.cuh"


namespace gtom
{
	class StackWarp2DOptimizer : public Optimizer
	{
	private:
		int2 dimsimage, dimsgrid;
		int2 dimsimagemasked;

		cudaArray_t* a_images;
		cudaTex* t_images;

		tfloat* h_gridx, *h_gridy;

		tfloat* d_prewarped;
		tfloat* d_buffersum, *d_buffercorr;
		tfloat* d_mask;

		tfloat* d_corrsum;

	public:
		int nimages;
		tfloat* h_scores;
		tfloat scoremean, scorestd;
		std::vector<uint> indexhold;

		StackWarp2DOptimizer(tfloat* _d_images,
			int2 _dimsimage,
			int _nimages,
			int2 _dimsgrid,
			std::vector<uint> &_indexhold,
			int _maxtrans)
		{
			nimages = _nimages;
			dimsimage = _dimsimage;
			dimsgrid = _dimsgrid;
			dimsimagemasked = toInt2(dimsimage.x - _maxtrans * 2, dimsimage.y - _maxtrans * 2);
			indexhold = _indexhold;

			cudaMalloc((void**)&d_prewarped, Elements2(dimsimage) * nimages * sizeof(tfloat));
			cudaMemcpy(d_prewarped, _d_images, Elements2(dimsimage) * nimages * sizeof(tfloat), cudaMemcpyDeviceToDevice);
			d_CubicBSplinePrefilter2D(d_prewarped, dimsimage, nimages);

			a_images = (cudaArray_t*)malloc(nimages * sizeof(cudaArray_t));
			t_images = (cudaTex*)malloc(nimages * sizeof(cudaTex));
			d_BindTextureToArray(d_prewarped, a_images, t_images, dimsimage, cudaFilterModeLinear, false, nimages);

			h_gridx = MallocValueFilled(Elements2(dimsgrid) * nimages, (tfloat)99999);
			h_gridy = MallocValueFilled(Elements2(dimsgrid) * nimages, (tfloat)99999);

			cudaMalloc((void**)&d_buffersum, Elements2(dimsimage) * sizeof(tfloat));
			cudaMalloc((void**)&d_buffercorr, Elements2(dimsimagemasked) * nimages * sizeof(tfloat));

			//d_mask = CudaMallocValueFilled(Elements2(dimsimage) * nimages, (tfloat)1);
			//d_RectangleMask(d_mask, d_mask, toInt3(dimsimage), toInt3(dimsimagemasked), NULL, nimages);
			//d_WriteMRC(d_mask, toInt3(dimsimage.x, dimsimage.y, nimages), "d_mask.mrc");

			cudaMalloc((void**)&d_corrsum, nimages *sizeof(tfloat));
			h_scores = (tfloat*)malloc(nimages * sizeof(tfloat));
		}

		~StackWarp2DOptimizer()
		{
			cudaFree(d_corrsum);
			cudaFree(d_mask);
			cudaFree(d_buffercorr);
			cudaFree(d_buffersum);
			cudaFree(d_prewarped);

			for (uint n = 0; n < nimages; n++)
			{
				cudaDestroyTextureObject(t_images[n]);
				cudaFreeArray(a_images[n]);
			}

			free(t_images);
			free(a_images);

			free(h_gridx);
			free(h_gridy);

			free(h_scores);
		}

		double Evaluate(column_vector& vec)
		{
			std::vector<int> needupdate;
			for (int n = 0; n < vec.size() / 2; n++)
			{
				bool keep = true;
				if (h_gridx[n] != (tfloat)vec(n * 2))
				{
					h_gridx[n] = vec(n * 2);
					keep = false;
				}
				if (h_gridy[n] != (tfloat)vec(n * 2 + 1))
				{
					h_gridy[n] = vec(n * 2 + 1);
					keep = false;
				}

				// If needs update and is not yet in needupdate
				if (!keep && std::find(needupdate.begin(), needupdate.end(), n / Elements2(dimsgrid)) == needupdate.end())
					needupdate.push_back(n / Elements2(dimsgrid));
			}

			for (uint i = 0; i < needupdate.size(); i++)
			{
				uint n = needupdate[i];

				tfloat* d_gridx = (tfloat*)CudaMallocFromHostArray(h_gridx + Elements2(dimsgrid) * n, Elements2(dimsgrid) * sizeof(tfloat));
				tfloat* d_gridy = (tfloat*)CudaMallocFromHostArray(h_gridy + Elements2(dimsgrid) * n, Elements2(dimsgrid) * sizeof(tfloat));
				d_CubicBSplinePrefilter2D(d_gridx, dimsgrid);
				d_CubicBSplinePrefilter2D(d_gridy, dimsgrid);
				cudaTex t_gridx, t_gridy;
				cudaArray_t a_gridx, a_gridy;
				d_BindTextureToArray(d_gridx, a_gridx, t_gridx, dimsgrid, cudaFilterModeLinear, false);
				d_BindTextureToArray(d_gridy, a_gridy, t_gridy, dimsgrid, cudaFilterModeLinear, false);
				cudaFree(d_gridy);
				cudaFree(d_gridx);

				d_Warp2D(t_images[n], dimsimage, t_gridx, t_gridy, dimsgrid, d_buffersum);
				d_Pad(d_buffersum, d_prewarped + Elements2(dimsimagemasked) * n, toInt3(dimsimage), toInt3(dimsimagemasked), T_PAD_VALUE, (tfloat)0);
				d_NormMonolithic(d_prewarped + Elements2(dimsimagemasked) * n, d_prewarped + Elements2(dimsimagemasked) * n, Elements2(dimsimagemasked), T_NORM_MEAN01STD, 1);

				cudaDestroyTextureObject(t_gridx);
				cudaDestroyTextureObject(t_gridy);
				cudaFreeArray(a_gridx);
				cudaFreeArray(a_gridy);
			}
			//d_WriteMRC(d_prewarped, toInt3(dimsimage.x, dimsimage.y, nimages), "d_prewarped.mrc");

			d_ReduceAdd(d_prewarped, d_buffersum, Elements2(dimsimagemasked), nimages);
			d_NormMonolithic(d_buffersum, d_buffersum, Elements2(dimsimagemasked), T_NORM_MEAN01STD, 1);
			//d_WriteMRC(d_buffersum, toInt3(dimsimage), "d_buffersum.mrc");

			d_MultiplyByVector(d_prewarped, d_buffersum, d_buffercorr, Elements2(dimsimagemasked), nimages);
			d_SumMonolithic(d_buffercorr, d_corrsum, Elements2(dimsimagemasked), nimages);
			cudaMemcpy(h_scores, d_corrsum, nimages * sizeof(tfloat), cudaMemcpyDeviceToHost);

			tfloat result = 0;
			scoremean = 0;
			scorestd = 0;
			for (int n = 0; n < nimages; n++)
			{
				h_scores[n] /= Elements2(dimsimagemasked);
				result += h_scores[n];
				scoremean += h_scores[n];
			}
			result /= nimages;
			result = 1.0 - result;

			scoremean /= nimages;
			for (int n = 0; n < nimages; n++)
				scorestd += (h_scores[n] - scoremean) * (h_scores[n] - scoremean);
			scorestd = sqrt(scorestd / (tfloat)nimages);

			return result * 10000.0;
		}

		void DumpIntermediate(column_vector& vec)
		{
			std::vector<int> needupdate;
			for (int n = 0; n < vec.size() / 2; n++)
			{
				bool keep = true;
				if (h_gridx[n] != (tfloat)vec(n * 2))
				{
					h_gridx[n] = vec(n * 2);
					keep = false;
				}
				if (h_gridy[n] != (tfloat)vec(n * 2 + 1))
				{
					h_gridy[n] = vec(n * 2 + 1);
					keep = false;
				}

				// If needs update and is not yet in needupdate
				if (!keep && std::find(needupdate.begin(), needupdate.end(), n / Elements2(dimsgrid)) == needupdate.end())
					needupdate.push_back(n / Elements2(dimsgrid));
			}

			for (uint i = 0; i < needupdate.size(); i++)
			{
				uint n = needupdate[i];

				tfloat* d_gridx = (tfloat*)CudaMallocFromHostArray(h_gridx + Elements2(dimsgrid) * n, Elements2(dimsgrid) * sizeof(tfloat));
				tfloat* d_gridy = (tfloat*)CudaMallocFromHostArray(h_gridy + Elements2(dimsgrid) * n, Elements2(dimsgrid) * sizeof(tfloat));
				d_CubicBSplinePrefilter2D(d_gridx, dimsgrid);
				d_CubicBSplinePrefilter2D(d_gridy, dimsgrid);
				cudaTex t_gridx, t_gridy;
				cudaArray_t a_gridx, a_gridy;
				d_BindTextureToArray(d_gridx, a_gridx, t_gridx, dimsgrid, cudaFilterModeLinear, false);
				d_BindTextureToArray(d_gridy, a_gridy, t_gridy, dimsgrid, cudaFilterModeLinear, false);
				cudaFree(d_gridy);
				cudaFree(d_gridx);

				d_Warp2D(t_images[n], dimsimage, t_gridx, t_gridy, dimsgrid, d_buffersum);
				d_Pad(d_buffersum, d_prewarped + Elements2(dimsimagemasked) * n, toInt3(dimsimage), toInt3(dimsimagemasked), T_PAD_VALUE, (tfloat)0);
				d_NormMonolithic(d_prewarped + Elements2(dimsimagemasked) * n, d_prewarped + Elements2(dimsimagemasked) * n, Elements2(dimsimagemasked), T_NORM_MEAN01STD, 1);

				cudaDestroyTextureObject(t_gridx);
				cudaDestroyTextureObject(t_gridy);
				cudaFreeArray(a_gridx);
				cudaFreeArray(a_gridy);
			}
			//d_WriteMRC(d_prewarped, toInt3(dimsimage.x, dimsimage.y, nimages), "d_prewarped.mrc");

			d_ReduceAdd(d_prewarped, d_buffersum, Elements2(dimsimagemasked), nimages);
			d_NormMonolithic(d_buffersum, d_buffersum, Elements2(dimsimagemasked), T_NORM_MEAN01STD, 1);
			d_WriteMRC(d_buffersum, toInt3(dimsimagemasked), "d_buffersum.mrc");
		}

		column_vector Derivative(column_vector x)
		{
			column_vector d = column_vector(x.size());
			for (int i = 0; i < x.size(); i++)
				d(i) = 0.0;

			for (int i = 0; i < nimages * Elements2(dimsgrid) * 2; i++)
			{
				if (std::find(indexhold.begin(), indexhold.end(), i / 2 / Elements2(dimsgrid)) != indexhold.end())
					continue;

				double old = x(i);

				x(i) = old + psi;
				double dp = this->Evaluate(x);
				x(i) = old - psi;
				double dn = this->Evaluate(x);

				x(i) = old;
				d(i) = (dp - dn) / (2.0 * psi);
			}

			DumpIntermediate(x);

			return d;
		}
	};

	void d_OptimizeStackWarp2D(tfloat* d_images,
		int2 dimsimage,
		int2 dimsgrid,
		uint nimages,
		uint indexhold,
		int maxshift,
		tfloat2* h_grids,
		tfloat* h_scores)
	{
		column_vector vec = column_vector(nimages * Elements2(dimsgrid) * 2);
		for (int i = 0; i < vec.size() / 2; i++)
		{
			vec(i * 2) = h_grids[i].x;
			vec(i * 2 + 1) = h_grids[i].y;
		}

		column_vector constraintlower = column_vector(vec.size());
		column_vector constraintupper = column_vector(vec.size());
		for (int n = 0; n < vec.size(); n++)
		{
			constraintlower(n) = n / 2 / Elements2(dimsgrid) == indexhold ? -1e-5 : -maxshift;
			constraintupper(n) = n / 2 / Elements2(dimsgrid) == indexhold ? 1e-5 : maxshift;
		}

		srand(123);

		std::vector<uint> indiceshold;
		indiceshold.push_back(indexhold);

		StackWarp2DOptimizer optimizer(d_images, dimsimage, nimages, dimsgrid, indiceshold, maxshift);
		optimizer.psi = 0.4;
		double initialscore = optimizer.Evaluate(vec);

		/*for (uint n = 2; n < nimages; n++)
		{
		optimizer.nimages = n;
		indiceshold.clear();
		indiceshold.push_back(indexhold);
		for (uint m = 0; m < n - 1; m++)
		indiceshold.push_back(m);
		optimizer.indexhold = indiceshold;

		dlib::find_min_box_constrained(dlib::bfgs_search_strategy(),
		dlib::objective_delta_stop_strategy(1.0),
		EvalWrapper(&optimizer),
		DerivativeWrapper(&optimizer),
		vec,
		constraintlower, constraintupper);
		}

		indiceshold.clear();
		indiceshold.push_back(indexhold);
		optimizer.indexhold = indiceshold;
		optimizer.nimages = nimages;*/

		dlib::find_min_box_constrained(dlib::bfgs_search_strategy(),
			dlib::objective_delta_stop_strategy(1e-1),
			EvalWrapper(&optimizer),
			DerivativeWrapper(&optimizer),
			vec,
			constraintlower, constraintupper);

		// Store results
		for (uint n = 0; n < vec.size() / 2; n++)
		{
			h_grids[n].x = vec(n * 2);
			h_grids[n].y = vec(n * 2 + 1);
		}
		for (uint n = 0; n < nimages; n++)
		{
			h_scores[n] = optimizer.h_scores[n];
		}
	}
}