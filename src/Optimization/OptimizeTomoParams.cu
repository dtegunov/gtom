#include "Prerequisites.cuh"
#include "Angles.cuh"
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


namespace gtom
{
	// CUDA kernel declarations:
	__global__ void BoundaryMaskKernel(tfloat* d_volume, int3 dimsvolume, int2 dimsimage, glm::mat4 transform);

	class TomoParamsSIRTOptimizer : public Optimizer
	{
	private:
		int nimages;
		int2 dimsimage;
		int3 dimsvolume;
		tfloat2 thetarange;
		tfloat maskradius;
		tfloat masksum;

		tfloat3* h_startangles;
		tfloat2* h_startshifts;
		tfloat2* h_startintensities;

		column_vector constraintlower;
		column_vector constraintupper;

		tfloat* d_images;

		tfloat* d_volume, *d_volumeresidue;

		tfloat* d_mask;
		tfloat* d_corrsum, *h_corrsum;

	public:
		TomoParamsSIRTOptimizer(tfloat* _d_images,
			int2 _dimsimage,
			int3 _dimsvolume,
			int _nimages,
			tfloat3* _h_startangles,
			tfloat2* _h_startshifts,
			tfloat2* _h_startintensities,
			column_vector _constraintlower,
			column_vector _constraintupper,
			tfloat* _d_mask)
		{
			dimsimage = _dimsimage;
			dimsvolume = _dimsvolume;
			nimages = _nimages;
			d_mask = _d_mask;

			h_startangles = _h_startangles;
			h_startshifts = _h_startshifts;
			h_startintensities = _h_startintensities;

			constraintlower = _constraintlower;
			constraintupper = _constraintupper;

			d_images = _d_images;

			cudaMalloc((void**)&d_volume, Elements(dimsvolume) * sizeof(tfloat));
			cudaMalloc((void**)&d_volumeresidue, Elements(dimsvolume) * sizeof(tfloat));

			cudaMalloc((void**)&d_corrsum, sizeof(tfloat));
			h_corrsum = (tfloat*)malloc(sizeof(tfloat));
		}

		~TomoParamsSIRTOptimizer()
		{
			cudaFree(d_volumeresidue);
			cudaFree(d_volume);
			cudaFree(d_corrsum);

			free(h_corrsum);
		}

		double Evaluate(column_vector& vec)
		{
			tfloat3* h_angles = (tfloat3*)malloc(nimages * sizeof(tfloat3));
			tfloat2* h_shifts = (tfloat2*)malloc(nimages * sizeof(tfloat2));
			tfloat2* h_intensities = (tfloat2*)malloc(nimages * sizeof(tfloat2));
			tfloat2 meanshift = tfloat2(0);
			for (int n = 0; n < nimages; n++)
			{
				h_angles[n] = tfloat3(h_startangles[n].x + ToRad(vec(n * 7)), h_startangles[n].y + ToRad(vec(n * 7 + 1)), h_startangles[n].z + ToRad(vec(n * 7 + 2)));
				h_shifts[n] = tfloat2(h_startshifts[n].x + vec(n * 7 + 3), h_startshifts[n].y + vec(n * 7 + 4));
				h_intensities[n] = tfloat2(h_startintensities[n].x + vec(n * 7 + 5) * 0.1, max(1e-4f, h_startintensities[n].y + vec(n * 7 + 6) * 0.1));
				meanshift.x += h_shifts[n].x;
				meanshift.y += h_shifts[n].y;
			}
			tfloat2* h_scales = (tfloat2*)MallocValueFilled(nimages * 2, (tfloat)1);
			meanshift.x /= (tfloat)nimages;
			meanshift.y /= (tfloat)nimages;
			/*for (int i = 0; i < nimages; i++)
			{
			h_shifts[i].x -= meanshift.x;
			h_shifts[i].y -= meanshift.y;
			}*/

			d_ValueFill(d_volume, Elements(dimsvolume), (tfloat)0);
			d_RecSIRT(d_volume, d_volumeresidue, dimsvolume, tfloat3(0, 0, 0), d_images, dimsimage, nimages, h_angles, h_shifts, h_scales, h_intensities, T_INTERP_CUBIC, 1, 50, true);
			//CudaWriteToBinaryFile("d_volume.bin", d_volumeresidue, Elements(dimsvolume) * sizeof(tfloat));
			d_SumMonolithic(d_volumeresidue, d_corrsum, d_mask, Elements(dimsvolume), 1);
			cudaMemcpy(h_corrsum, d_corrsum, sizeof(tfloat), cudaMemcpyDeviceToHost);

			free(h_scales);
			free(h_intensities);
			free(h_shifts);
			free(h_angles);

			return h_corrsum[0] * 1000000.0;
		}

		void DumpRec(column_vector& vec)
		{
			tfloat3* h_angles = (tfloat3*)malloc(nimages * sizeof(tfloat3));
			tfloat2* h_shifts = (tfloat2*)malloc(nimages * sizeof(tfloat2));
			tfloat2* h_intensities = (tfloat2*)malloc(nimages * sizeof(tfloat2));
			tfloat2 meanshift = tfloat2(0);
			for (int n = 0; n < nimages; n++)
			{
				h_angles[n] = tfloat3(h_startangles[n].x + ToRad(vec(n * 7)), h_startangles[n].y + ToRad(vec(n * 7 + 1)), h_startangles[n].z + ToRad(vec(n * 7 + 2)));
				h_shifts[n] = tfloat2(h_startshifts[n].x + vec(n * 7 + 3), h_startshifts[n].y + vec(n * 7 + 4));
				h_intensities[n] = tfloat2(h_startintensities[n].x + vec(n * 7 + 5) * 0.1, max(1e-4f, h_startintensities[n].y + vec(n * 7 + 6) * 0.1));
				meanshift.x += h_shifts[n].x;
				meanshift.y += h_shifts[n].y;
			}
			tfloat2* h_scales = (tfloat2*)MallocValueFilled(nimages * 2, (tfloat)1);
			meanshift.x /= (tfloat)nimages;
			meanshift.y /= (tfloat)nimages;
			/*for (int i = 0; i < nimages; i++)
			{
			h_shifts[i].x -= meanshift.x;
			h_shifts[i].y -= meanshift.y;
			}*/

			d_ValueFill(d_volume, Elements(dimsvolume), (tfloat)0);
			d_RecSIRT(d_volume, d_volumeresidue, dimsvolume, tfloat3(0, 0, 0), d_images, dimsimage, nimages, h_angles, h_shifts, h_scales, h_intensities, T_INTERP_CUBIC, 1, 100, true);
			CudaWriteToBinaryFile("d_volume.bin", d_volume, Elements(dimsvolume) * sizeof(tfloat));

			free(h_scales);
			free(h_intensities);
			free(h_shifts);
			free(h_angles);
		}

		column_vector Derivative(column_vector x)
		{
			double componentpsi[7];
			componentpsi[0] = 0.2;
			componentpsi[1] = 0.2;
			componentpsi[2] = 0.2;
			componentpsi[3] = 0.2;
			componentpsi[4] = 0.2;
			componentpsi[5] = 0.5;
			componentpsi[6] = 0.5;

			column_vector d = column_vector(x.size());
			for (int i = 0; i < x.size(); i++)
				d(i) = 0.0;

			for (int i = 0; i < x.size(); i++)
			{
				if (abs(constraintupper(i) - constraintlower(i)) < 1e-8)
					continue;

				double old = x(i);

				x(i) = old + componentpsi[i % 7];
				double dp = this->Evaluate(x);
				x(i) = old - componentpsi[i % 7];
				double dn = this->Evaluate(x);

				x(i) = old;
				d(i) = (dp - dn) / (2.0 * componentpsi[i % 7]);
			}
			DumpRec(x);

			return d;
		}
	};

	__global__ void BoundaryMaskKernel(tfloat* d_volume, int3 dimsvolume, int2 dimsimage, glm::mat4 transform)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= dimsvolume.x)
			return;
		int idy = blockIdx.y * blockDim.y + threadIdx.y;
		if (idy >= dimsvolume.y)
			return;
		int idz = blockIdx.z;
		int outx = idx;
		int outy = idy;
		int outz = idz;

		glm::vec4 position = glm::vec4(idx, idy, idz, 1);
		position = transform * position;

		if (position.x < -1e-8f || position.x >= dimsimage.x + 1e-8f || position.y < -1e-8f || position.y >= dimsimage.y + 1e-8f)
			d_volume[(outz * dimsvolume.y + outy) * dimsvolume.x + outx] = (tfloat)0;
	}

	void d_OptimizeTomoParams(tfloat* d_images,
		int2 dimsimage,
		int3 dimsvolume,
		int nimages,
		std::vector<int> indiceshold,
		tfloat3* h_angles,
		tfloat2* h_shifts,
		tfloat2* h_intensities,
		tfloat3 deltaanglesmax,
		tfloat deltashiftmax,
		tfloat2 deltaintensitymax,
		tfloat &finalscore)
	{
		column_vector vec = column_vector(nimages * 7);
		for (int i = 0; i < vec.size(); i++)
			vec(i) = 0.0;

		column_vector constraintlower = column_vector(nimages * 7);
		column_vector constraintupper = column_vector(nimages * 7);
		for (int n = 0; n < nimages; n++)
		{
			bool hold = std::find(indiceshold.begin(), indiceshold.end(), n) != indiceshold.end();

			constraintlower(n * 7) = -(hold ? 1e-10 : ToDeg(deltaanglesmax.x));
			constraintlower(n * 7 + 1) = -(hold ? 1e-10 : ToDeg(deltaanglesmax.y));
			constraintlower(n * 7 + 2) = -(hold ? 1e-10 : ToDeg(deltaanglesmax.z));
			constraintlower(n * 7 + 3) = -(hold ? 1e-10 : deltashiftmax);
			constraintlower(n * 7 + 4) = -(hold ? 1e-10 : deltashiftmax);
			constraintlower(n * 7 + 5) = -(hold ? 1e-10 : deltaintensitymax.x);
			constraintlower(n * 7 + 6) = -(hold ? 1e-10 : deltaintensitymax.y);

			constraintupper(n * 7) = hold ? 1e-10 : ToDeg(deltaanglesmax.x);
			constraintupper(n * 7 + 1) = hold ? 1e-10 : ToDeg(deltaanglesmax.y);
			constraintupper(n * 7 + 2) = hold ? 1e-10 : ToDeg(deltaanglesmax.z);
			constraintupper(n * 7 + 3) = (hold ? 1e-10 : deltashiftmax);
			constraintupper(n * 7 + 4) = (hold ? 1e-10 : deltashiftmax);
			constraintupper(n * 7 + 5) = (hold ? 1e-10 : deltaintensitymax.x);
			constraintupper(n * 7 + 6) = (hold ? 1e-10 : deltaintensitymax.y);
		}

		// Estimate minimum volume that covers all voxels described by at least one image
		int3 dimsvolumeextended;
		{
			float maxx = -1e30f, maxy = -1e30f;
			float minx = 1e30f, miny = 1e30f;

			for (int i = 0; i < nimages; i++)
				for (float mx = -deltashiftmax; mx <= deltashiftmax + 1e-8f; mx += max(deltashiftmax, 1.0f))
					for (float my = -deltashiftmax; my <= deltashiftmax + 1e-8f; my += max(deltashiftmax, 1.0f))
						for (float mphi = -deltaanglesmax.x; mphi <= deltaanglesmax.x + 1e-8f; mphi += max(deltaanglesmax.x / 4.0f, ToRad(0.5f)))
							for (float mtheta = -deltaanglesmax.y; mtheta <= deltaanglesmax.y + 1e-8f; mtheta += max(deltaanglesmax.y / 4.0f, ToRad(0.5f)))
								for (float mpsi = -deltaanglesmax.z; mpsi <= deltaanglesmax.z + 1e-8f; mpsi += max(deltaanglesmax.z / 4.0f, ToRad(0.5f)))
								{
									glm::mat4 transform = Matrix4Translation(tfloat3(dimsimage.x / 2, dimsimage.y / 2, 0)) *
										Matrix4Euler(tfloat3(h_angles[i].x + mphi, h_angles[i].y + mtheta, h_angles[i].z + mpsi)) *
										Matrix4Translation(tfloat3(-dimsvolume.x / 2, -dimsvolume.y / 2, 0));

									glm::vec4 ray = Matrix4Euler(tfloat3(h_angles[i].x + mphi, h_angles[i].y + mtheta, h_angles[i].z + mpsi)) * glm::vec4(0.0f, 0.0f, -1.0f, 1.0f);
									glm::vec4 corners[4];
									corners[0] = transform * glm::vec4(0, 0, 0, 1);
									corners[1] = transform * glm::vec4(0, dimsimage.y, 0, 1);
									corners[2] = transform * glm::vec4(dimsimage.x, 0, 0, 1);
									corners[3] = transform * glm::vec4(dimsimage.x, dimsimage.y, 0, 1);

									for (int c = 0; c < 4; c++)
									{
										float toupperplane = (corners[c].z - (float)(dimsvolume.z / 2)) / ray.z;
										glm::vec4 intersect1 = corners[c] - ray * toupperplane;
										float tobottomplane = (corners[c].z + (float)(dimsvolume.z / 2)) / ray.z;
										glm::vec4 intersect2 = corners[c] - ray * tobottomplane;
										maxx = max(maxx, max(intersect1.x, intersect2.x));
										maxy = max(maxy, max(intersect1.y, intersect2.y));
										minx = min(minx, min(intersect1.x, intersect2.x));
										miny = min(miny, min(intersect1.y, intersect2.y));
									}
								}

			dimsvolumeextended = toInt3(ceil(maxx) - floor(minx), ceil(maxy) - floor(miny), dimsvolume.z);
			dimsvolumeextended.x += dimsvolumeextended.x % 2;
			dimsvolumeextended.y += dimsvolumeextended.y % 2;
		}

		// Create volume mask that marks positions covered by all images 
		// under all parameter combinations within the given constraints
		tfloat* d_mask = CudaMallocValueFilled(Elements(dimsvolumeextended), (tfloat)0);
		{
			for (int i = 0; i < nimages; i++)
			{
				tfloat* d_masktemp = CudaMallocValueFilled(Elements(dimsvolumeextended), (tfloat)1);

				for (float mx = -deltashiftmax; mx <= deltashiftmax + 1e-8f; mx += max(deltashiftmax, 1.0f))
					for (float my = -deltashiftmax; my <= deltashiftmax + 1e-8f; my += max(deltashiftmax, 1.0f))
						for (float mphi = -deltaanglesmax.x; mphi <= deltaanglesmax.x + 1e-8f; mphi += max(deltaanglesmax.x / 4.0f, ToRad(0.5f)))
							for (float mtheta = -deltaanglesmax.y; mtheta <= deltaanglesmax.y + 1e-8f; mtheta += max(deltaanglesmax.y / 4.0f, ToRad(0.5f)))
								for (float mpsi = -deltaanglesmax.z; mpsi <= deltaanglesmax.z + 1e-8f; mpsi += max(deltaanglesmax.z / 4.0f, ToRad(0.5f)))
								{
									glm::mat4 transform = Matrix4Translation(tfloat3(dimsimage.x / 2 + h_shifts[i].x + mx, dimsimage.y / 2 + h_shifts[i].y + my, 0.0f)) *
										glm::transpose(Matrix4Euler(tfloat3(h_angles[i].x + mphi, h_angles[i].y + mtheta, h_angles[i].z + mpsi))) *
										Matrix4Translation(tfloat3(-dimsvolumeextended.x / 2, -dimsvolumeextended.y / 2, -dimsvolumeextended.z / 2));
									dim3 TpB = dim3(16, 16);
									dim3 grid = dim3((dimsvolumeextended.x + 15) / 16, (dimsvolumeextended.y + 15) / 16, dimsvolumeextended.z);

									BoundaryMaskKernel << <grid, TpB >> > (d_masktemp, dimsvolumeextended, dimsimage, transform);
								}

				d_AddVector(d_mask, d_masktemp, d_mask, Elements(dimsvolumeextended));
				cudaFree(d_masktemp);
			}

			d_MaxOp(d_mask, (tfloat)(nimages - 2), d_mask, Elements(dimsvolumeextended));
			d_SubtractScalar(d_mask, d_mask, Elements(dimsvolumeextended), (tfloat)(nimages - 2));
			d_Sqrt(d_mask, d_mask, Elements(dimsvolumeextended));

			tfloat* d_masksum = CudaMallocValueFilled(1, (tfloat)0);
			d_Sum(d_mask, d_masksum, Elements(dimsvolumeextended));
			tfloat masksum = 0;
			cudaMemcpy(&masksum, d_masksum, sizeof(tfloat), cudaMemcpyDeviceToHost);
			cudaFree(d_masksum);
			if (masksum == 0)
				throw;
			d_DivideByScalar(d_mask, d_mask, Elements(dimsvolumeextended), masksum);

			CudaWriteToBinaryFile("d_mask.bin", d_mask, Elements(dimsvolumeextended) * sizeof(tfloat));
		}

		TomoParamsSIRTOptimizer optimizer(d_images, dimsimage, dimsvolumeextended, nimages, h_angles, h_shifts, h_intensities, constraintlower, constraintupper, d_mask);

		double initialscore = optimizer.Evaluate(vec);

		optimizer.psi = 1.0;
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
			h_angles[n] = tfloat3(h_angles[n].x + ToRad(vec(n * 7)), h_angles[n].y + ToRad(vec(n * 7 + 1)), h_angles[n].z + ToRad(vec(n * 7 + 2)));
			h_shifts[n] = tfloat2(h_shifts[n].x + vec(n * 7 + 3), h_shifts[n].y + vec(n * 7 + 4));
			h_intensities[n] = tfloat2(h_intensities[n].x + vec(n * 7 + 5) * 0.1, h_intensities[n].y + vec(n * 7 + 6) * 0.1);
		}

		cudaFree(d_mask);
	}
}