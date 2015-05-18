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


// CUDA kernel declarations:
__global__ void BoundaryMaskWBPKernel(tfloat* d_volume, int3 dimsvolume, int2 dimsimage, glm::mat4 transform);

class TomoParamsWBPOptimizer : public Optimizer
{
private:
	int nimages;
	int ngroups;
	vector<vector<int>> indicesgroups;
	int2 dimsimage, dimsimageoriginal, dimscropped;
	vector<int3> dimsvolume, dimsvolumeoriginal;
	tfloat scalefactor;
	tfloat2 thetarange;
	tfloat maskradius;
	vector<tfloat> masksum;

	tfloat* d_images;
	vector<tcomplex*> v_imagesft;
	tcomplex* d_imagesft2;
	vector<tfloat*> v_imagesnorm;

	tfloat* d_volume;
	tfloat* d_proj;

	vector<tfloat*> v_mask, v_maskoriginal;
	tfloat* d_corrsum, *h_corrsum;
	vector<tfloat*> v_masksums;

	cufftHandle planback;

	bool initialized;

public:
	tfloat* h_scores;
	tfloat scoremean, scorestd;
	tfloat freqlow, freqhigh;

	column_vector constraintlower;
	column_vector constraintupper;

	TomoParamsWBPOptimizer(tfloat* _d_images,
		int2 _dimsimage,
		vector<int3> &_dimsvolume,
		int _nimages,
		vector<vector<int>> &_indicesgroups,
		tfloat _freqlow, tfloat _freqhigh,
		column_vector _constraintlower,
		column_vector _constraintupper,
		vector<tfloat*> &_v_mask)
	{
		initialized = false;

		nimages = _nimages;
		ngroups = _indicesgroups.size();
		indicesgroups = _indicesgroups;
		dimsimageoriginal = _dimsimage;
		dimsvolumeoriginal = _dimsvolume;
		dimsvolume = vector<int3>(ngroups);
		freqlow = _freqlow;
		freqhigh = _freqhigh;

		v_maskoriginal = _v_mask;

		constraintlower = _constraintlower;
		constraintupper = _constraintupper;

		d_images = _d_images;

		Reinitialize();
	}

	void Reinitialize()
	{
		Deinitialize();

		// Scale all sizes to fit current highest frequency
		scalefactor = freqhigh / (tfloat)(dimsimageoriginal.x / 2);
		dimsimage = toInt2(dimsimageoriginal.x * scalefactor, dimsimageoriginal.y * scalefactor);
		int3 dimsvolumemax = toInt3(1, 1, 1);
		int pergroupmax = 1;
		for (uint g = 0; g < ngroups; g++)
		{
			dimsvolume[g] = toInt3(dimsvolumeoriginal[g].x * scalefactor, dimsvolumeoriginal[g].y * scalefactor, dimsvolumeoriginal[g].z * scalefactor);
			dimsvolumemax = toInt3(max(dimsvolumemax.x, dimsvolume[g].x), max(dimsvolumemax.y, dimsvolume[g].y), max(dimsvolumemax.z, dimsvolume[g].z));
			pergroupmax = max(pergroupmax, (int)indicesgroups[g].size());
		}
		dimscropped = toInt2(dimsimage.x * 7 / 8, dimsimage.y * 7 / 8);

		// Allocate each group's buffers
		for (uint g = 0; g < ngroups; g++)
		{
			int groupsize = indicesgroups[g].size();

			// Make it pergroupmax so one IFFT plan can suit all needs later
			tcomplex* d_imagesft;
			cudaMalloc((void**)&d_imagesft, ElementsFFT2(dimsimage) * pergroupmax * sizeof(tcomplex));
			v_imagesft.push_back(d_imagesft);

			tfloat* d_imagesnorm;
			cudaMalloc((void**)&d_imagesnorm, Elements2(dimscropped) * groupsize * sizeof(tfloat));
			v_imagesnorm.push_back(d_imagesnorm);

			tfloat* d_mask;
			cudaMalloc((void**)&d_mask, Elements2(dimscropped) * groupsize * sizeof(tfloat));
			v_mask.push_back(d_mask);
		}

		cudaMalloc((void**)&d_imagesft2, ElementsFFT2(dimsimageoriginal) * nimages * sizeof(tcomplex));
		
		cudaMalloc((void**)&d_volume, Elements(dimsvolumemax) * sizeof(tfloat));
		cudaMalloc((void**)&d_proj, Elements2(dimsimage) * nimages * sizeof(tfloat));

		cudaMalloc((void**)&d_corrsum, nimages * sizeof(tfloat));
		h_corrsum = (tfloat*)malloc(nimages * sizeof(tfloat));

		// Resize each group's masks and count samples
		masksum.clear();
		for (uint g = 0; g < ngroups; g++)
		{
			d_Scale(v_maskoriginal[g], (tfloat*)d_imagesft2, toInt3(dimsimageoriginal), toInt3(dimsimage), T_INTERP_CUBIC, NULL, NULL, indicesgroups[g].size());
			d_Pad((tfloat*)d_imagesft2, v_mask[g], toInt3(dimsimage), toInt3(dimscropped), T_PAD_VALUE, (tfloat)0, indicesgroups[g].size());
			d_Binarize(v_mask[g], v_mask[g], Elements2(dimscropped) * indicesgroups[g].size(), 0.8);
			d_WriteMRC(v_mask[g], toInt3(dimscropped.x, dimscropped.y, indicesgroups[g].size()), "d_mask" + to_string(g) + ".mrc");

			d_SumMonolithic(v_mask[g], d_corrsum, Elements2(dimscropped), indicesgroups[g].size());
			tfloat* h_masksum = (tfloat*)MallocFromDeviceArray(d_corrsum, indicesgroups[g].size() * sizeof(tfloat));
			tfloat masksumtemp = 0;
			for (int n = 0; n < indicesgroups[g].size(); n++)
				masksumtemp += h_masksum[n];
			masksum.push_back(masksumtemp);
			v_masksums.push_back(h_masksum);
		}

		h_scores = (tfloat*)malloc(nimages * sizeof(tfloat));

		// Pre-FT all images, and bandpass them to current limits
		d_FFTR2C(d_images, d_imagesft2, 2, toInt3(dimsimageoriginal), nimages);
		d_FFTCrop(d_imagesft2, d_imagesft2, toInt3(dimsimageoriginal), toInt3(dimsimage), nimages);
		d_Bandpass(d_imagesft2, d_imagesft2, toInt3(dimsimage), freqlow, freqhigh, 0, NULL, nimages);

		tcomplex* d_buffer;
		cudaMalloc((void**)&d_buffer, ElementsFFT2(dimsimage) * nimages * sizeof(tcomplex));

		// Copy FTs from specified indices to the group's buffer
		for (uint g = 0; g < ngroups; g++)
		{
			for (uint n = 0; n < indicesgroups[g].size(); n++)
				cudaMemcpy(v_imagesft[g] + n * ElementsFFT2(dimsimage), d_imagesft2 + indicesgroups[g][n] * ElementsFFT2(dimsimage), ElementsFFT2(dimsimage) * sizeof(tcomplex), cudaMemcpyDeviceToDevice);
		}

		// Transform each group back, crop and normalize
		for (uint g = 0; g < ngroups; g++)
		{
			d_IFFTC2R(v_imagesft[g], (tfloat*)d_buffer, 2, toInt3(dimsimage), indicesgroups[g].size(), false);
			d_WriteMRC((tfloat*)d_buffer, toInt3(dimsimage.x, dimsimage.y, indicesgroups[g].size()), "d_images" + to_string(g) + ".mrc");

			d_Pad((tfloat*)d_buffer, v_imagesnorm[g], toInt3(dimsimage), toInt3(dimscropped), T_PAD_VALUE, (tfloat)0, indicesgroups[g].size());
			d_NormMonolithic(v_imagesnorm[g], v_imagesnorm[g], Elements2(dimscropped), v_mask[g], T_NORM_MEAN01STD, indicesgroups[g].size());
			d_WriteMRC(v_imagesnorm[g], toInt3(dimscropped.x, dimscropped.y, indicesgroups[g].size()), "d_imagesnorm" + to_string(g) + ".mrc");
		}

		cudaFree(d_buffer);

		// Buffers involved in this IFFT are all pergroupmax big
		planback = d_IFFTC2RGetPlan(2, toInt3(dimsimage), pergroupmax);

		initialized = true;
	}

	void Deinitialize()
	{
		if (initialized)
		{
			cufftDestroy(planback);

			for (uint g = 0; g < ngroups; g++)
			{
				cudaFree(v_imagesft[g]);
				cudaFree(v_imagesnorm[g]);
				cudaFree(v_mask[g]);
				free(v_masksums[g]);
			}
			v_imagesft.clear();
			v_imagesnorm.clear();
			v_mask.clear();
			v_masksums.clear();

			cudaFree(d_imagesft2);
			cudaFree(d_volume);
			cudaFree(d_proj);
			cudaFree(d_corrsum);

			free(h_corrsum);
			free(h_scores);

			initialized = false;
		}
	}

	~TomoParamsWBPOptimizer()
	{
		Deinitialize();
	}

	double Evaluate(column_vector& vec)
	{
		for (uint g = 0; g < ngroups; g++)
		{
			uint groupsize = indicesgroups[g].size();

			tfloat3* h_angles = (tfloat3*)malloc(groupsize * sizeof(tfloat3));
			tfloat2* h_shifts = (tfloat2*)malloc(groupsize * sizeof(tfloat2));
			for (int m = 0; m < groupsize; m++)
			{
				uint n = indicesgroups[g][m];
				h_angles[m] = tfloat3(ToRad(vec(n * 5)) * 0.1f, ToRad(vec(n * 5 + 1)), ToRad(vec(n * 5 + 2)));
				h_shifts[m] = tfloat2(vec(n * 5 + 3) * scalefactor, vec(n * 5 + 4) * scalefactor);
			}
			tfloat2* h_scales = (tfloat2*)MallocValueFilled(groupsize * 2, (tfloat)1);

			int* h_indices = (int*)malloc(nimages * sizeof(int));
			for (int n = 0; n < nimages; n++)
				h_indices[n] = n;
			cudaMemcpy(d_imagesft2, d_imagesft, ElementsFFT2(dimsimage) * nimages * sizeof(tcomplex), cudaMemcpyDeviceToDevice);
			d_Exact2DWeighting(d_imagesft2, dimsimage, h_indices, h_angles, nimages, dimsimage.x, false, nimages);
			free(h_indices);

			d_IFFTC2R(d_imagesft2, d_proj, &planback);

			d_ValueFill(d_volume, Elements(dimsvolume), (tfloat)0);
			d_ProjBackward(d_volume, dimsvolume, tfloat3(0), d_proj, dimsimage, h_angles, h_shifts, h_scales, T_INTERP_CUBIC, true, nimages);
			//d_WriteMRC(d_volume, dimsvolume, "d_volume.mrc");

			d_ProjForwardRaytrace(d_volume, dimsvolume, d_proj, dimscropped, h_angles, h_shifts, h_scales, T_INTERP_CUBIC, 1, nimages);
			d_NormMonolithic(d_proj, d_proj, Elements2(dimscropped), d_mask, T_NORM_MEAN01STD, nimages);
			//d_WriteMRC(d_proj, toInt3(dimsimage.x, dimsimage.y, nimages), "d_proj.mrc");

			d_MultiplyByVector(d_proj, d_imagesnorm, d_proj, Elements2(dimscropped) * nimages);
			d_SumMonolithic(d_proj, d_corrsum, d_mask, Elements2(dimscropped), nimages);
			cudaMemcpy(h_corrsum, d_corrsum, nimages * sizeof(tfloat), cudaMemcpyDeviceToHost);

			tfloat result = 0;
			scoremean = 0;
			scorestd = 0;
			for (int n = 0; n < nimages; n++)
			{
				result += h_corrsum[n];
				h_scores[n] = h_corrsum[n] / h_masksum[n];
				scoremean += h_scores[n];
			}
			result /= masksum;
			result = 1.0 - result;
		}

		scoremean /= (tfloat)nimages;
		for (int n = 0; n < nimages; n++)
			scorestd += (h_scores[n] - scoremean) * (h_scores[n] - scoremean);
		scorestd = sqrt(scorestd / (tfloat)nimages);

		free(h_scales);
		free(h_shifts);
		free(h_angles);

		return result * 10000.0;
	}

	void DumpRec(column_vector& vec)
	{
		/*tfloat3* h_angles = (tfloat3*)malloc(nimages * sizeof(tfloat3));
		tfloat2* h_shifts = (tfloat2*)malloc(nimages * sizeof(tfloat2));
		for (int n = 0; n < nimages; n++)
		{
			h_angles[n] = tfloat3(ToRad(vec(n * 5)) * 0.1f, ToRad(vec(n * 5 + 1)), ToRad(vec(n * 5 + 2)));
			h_shifts[n] = tfloat2(vec(n * 5 + 3) * scalefactor, vec(n * 5 + 4) * scalefactor);
		}
		tfloat2* h_scales = (tfloat2*)MallocValueFilled(nimages * 2, (tfloat)1);

		int* h_indices = (int*)malloc(nimages * sizeof(int));
		for (int n = 0; n < nimages; n++)
			h_indices[n] = n;
		cudaMemcpy(d_imagesft2, d_imagesft, ElementsFFT2(dimsimage) * nimages * sizeof(tcomplex), cudaMemcpyDeviceToDevice);
		d_Exact2DWeighting(d_imagesft2, dimsimage, h_indices, h_angles, nimages, dimsimage.x, false, nimages);
		free(h_indices);

		d_IFFTC2R(d_imagesft2, d_proj, &planback);

		d_ValueFill(d_volume, Elements(dimsvolume), (tfloat)0);
		d_ProjBackward(d_volume, dimsvolume, tfloat3(0), d_proj, dimsimage, h_angles, h_shifts, h_scales, T_INTERP_CUBIC, true, nimages);
		d_WriteMRC(d_volume, dimsvolume, "d_volume.mrc");

		d_ProjForwardRaytrace(d_volume, dimsvolume, d_proj, dimscropped, h_angles, h_shifts, h_scales, T_INTERP_CUBIC, 1, nimages);
		d_NormMonolithic(d_proj, d_proj, Elements2(dimscropped), d_mask, T_NORM_MEAN01STD, nimages);
		d_WriteMRC(d_proj, toInt3(dimscropped.x, dimscropped.y, nimages), "d_proj.mrc");

		free(h_scales);
		free(h_shifts);
		free(h_angles);*/
	}

	column_vector Derivative(column_vector x)
	{
		/*double componentpsi[5];
		componentpsi[0] = 1.0;
		componentpsi[1] = 1.0;
		componentpsi[2] = 1.0;
		componentpsi[3] = 1.0;
		componentpsi[4] = 1.0;*/

		column_vector d = column_vector(x.size());
		for (int i = 0; i < x.size(); i++)
			d(i) = 0.0;

		for (int i = 0; i < x.size(); i++)
		{
			if (abs(constraintupper(i) - constraintlower(i)) < 1e-3)
				continue;

			double old = x(i);

			x(i) = old + psi;
			double dp = this->Evaluate(x);
			x(i) = old - psi;
			double dn = this->Evaluate(x);

			x(i) = old;
			d(i) = (dp - dn) / (2.0 * psi);
		}
		DumpRec(x);

		return d;
	}
};

__global__ void BoundaryMaskWBPKernel(tfloat* d_volume, int3 dimsvolume, int2 dimsimage, glm::mat4 transform)
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

void d_OptimizeTomoParamsWBP(tfloat* d_images,
	int2 dimsimage,
	vector<int3> &dimsvolume,
	int nimages,
	vector<int> &indiceshold,
	vector<vector<int>> &indicesgroups,
	tfloat3* h_angles,
	tfloat2* h_shifts,
	tfloat3* h_deltaanglesmin, tfloat3* h_deltaanglesmax,
	tfloat2* h_deltashiftmin, tfloat2* h_deltashiftmax,
	tfloat &finalscore)
{
	uint ngroups = indicesgroups.size();

	column_vector vec = column_vector(nimages * 5);
	for (int i = 0; i < nimages; i++)
	{
		vec(i * 5 + 0) = h_angles[i].x;
		vec(i * 5 + 1) = h_angles[i].y;
		vec(i * 5 + 2) = h_angles[i].z;
		vec(i * 5 + 3) = h_shifts[i].x;
		vec(i * 5 + 4) = h_shifts[i].y;
	}

	column_vector constraintlower = column_vector(nimages * 5);
	column_vector constraintupper = column_vector(nimages * 5);
	for (int n = 0; n < nimages; n++)
	{
		bool hold = std::find(indiceshold.begin(), indiceshold.end(), n) != indiceshold.end();
		bool holdphi = abs(h_angles[n].y) < 1e-3;

		constraintlower(n * 5) = holdphi ? h_angles[n].x - ToRad(1e-5) : h_deltaanglesmin[n].x;
		constraintlower(n * 5 + 1) = hold ? h_angles[n].y - ToRad(1e-5) : h_deltaanglesmin[n].y;
		constraintlower(n * 5 + 2) = h_deltaanglesmin[n].z;
		constraintlower(n * 5 + 3) = hold ? h_shifts[n].x - 1e-5 : h_deltashiftmin[n].x;
		constraintlower(n * 5 + 4) = hold ? h_shifts[n].y - 1e-5 : h_deltashiftmin[n].y;

		constraintupper(n * 5) = holdphi ? h_angles[n].x + ToRad(1e-5) : h_deltaanglesmax[n].x;
		constraintupper(n * 5 + 1) = hold ? h_angles[n].y + ToRad(1e-5) : h_deltaanglesmax[n].y;
		constraintupper(n * 5 + 2) = h_deltaanglesmax[n].z;
		constraintupper(n * 5 + 3) = hold ? h_shifts[n].x + 1e-5 : h_deltashiftmax[n].x;
		constraintupper(n * 5 + 4) = hold ? h_shifts[n].y + 1e-5 : h_deltashiftmax[n].y;
	}

	// Create volume masks that mark positions covered by all images of a group
	// under all parameter combinations within the given constraints
	vector<tfloat*> v_masks;
	for (uint g = 0; g < ngroups; g++)
	{
		tfloat* d_mask;
		cudaMalloc((void**)&d_mask, Elements2(dimsimage) * indicesgroups[g].size() * sizeof(tfloat));
		{
			tfloat* d_maskvolume = CudaMallocValueFilled(Elements(dimsvolume[g]), (tfloat)1);
			for (int j = 0; j < indicesgroups[g].size(); j++)
			{
				int i = indicesgroups[g][j];

				for (float mx = constraintlower(i * 5 + 3); mx <= constraintupper(i * 5 + 3) + 1e-11f; mx += max((h_deltashiftmax[i].x - h_deltashiftmin[i].x) / 2.0f, 1.0f))
					for (float my = constraintlower(i * 5 + 4); my <= constraintupper(i * 5 + 4) + 1e-11f; my += max((h_deltashiftmax[i].y - h_deltashiftmin[i].y) / 2.0f, 1.0f))
						for (float mphi = constraintlower(i * 5 + 0); mphi <= constraintupper(i * 5 + 0) + 1e-11f; mphi += max((h_deltaanglesmax[i].x - h_deltaanglesmin[i].x) / 8.0f, ToRad(0.5f)))
							for (float mtheta = constraintlower(i * 5 + 1); mtheta <= constraintupper(i * 5 + 1); mtheta += max((h_deltaanglesmax[i].y - h_deltaanglesmin[i].y) / 8.0f, ToRad(0.5f)))
								for (float mpsi = constraintlower(i * 5 + 2); mpsi <= constraintupper(i * 5 + 2) + 1e-11f; mpsi += max((h_deltaanglesmax[i].z - h_deltaanglesmin[i].z) / 8.0f, ToRad(0.5f)))
								{
									glm::mat4 transform = Matrix4Translation(tfloat3(dimsimage.x / 2 + mx, dimsimage.y / 2 + my, 0.0f)) *
										glm::transpose(Matrix4Euler(tfloat3(mphi, mtheta, mpsi))) *
										Matrix4Translation(tfloat3(-dimsvolume[g].x / 2, -dimsvolume[g].y / 2, -dimsvolume[g].z / 2));

									dim3 TpB = dim3(16, 16);
									dim3 grid = dim3((dimsvolume[g].x + 15) / 16, (dimsvolume[g].y + 15) / 16, dimsvolume[g].z);

									BoundaryMaskWBPKernel << <grid, TpB >> > (d_maskvolume, dimsvolume[g], dimsimage, transform);
								}
			}

			{
				tfloat3* h_groupangles = (tfloat3*)malloc(indicesgroups[g].size() * sizeof(tfloat3));
				tfloat2* h_groupshifts = (tfloat2*)malloc(indicesgroups[g].size() * sizeof(tfloat2));
				for (uint n = 0; n < indicesgroups[g].size(); n++)
				{
					h_groupangles[n] = h_angles[indicesgroups[g][n]];
					h_groupshifts[n] = h_shifts[indicesgroups[g][n]];
				}
				tfloat2* h_shiftszero = (tfloat2*)MallocValueFilled(indicesgroups[g].size() * 2, (tfloat)0);
				tfloat2* h_scales = (tfloat2*)MallocValueFilled(indicesgroups[g].size() * 2, (tfloat)1);
				d_ProjForwardRaytrace(d_maskvolume, dimsvolume[g], d_mask, dimsimage, h_groupangles, h_groupshifts, h_scales, T_INTERP_LINEAR, 1, indicesgroups[g].size());
				free(h_groupangles);
				free(h_groupshifts);
				free(h_shiftszero);
				free(h_scales);
			}

			tfloat* d_maxvals = CudaMallocValueFilled(indicesgroups[g].size(), (tfloat)0);
			d_MaxMonolithic(d_mask, d_maxvals, Elements2(dimsimage), indicesgroups[g].size());
			tfloat* h_maxvals = (tfloat*)MallocFromDeviceArray(d_maxvals, indicesgroups[g].size() * sizeof(tfloat));
			cudaFree(d_maxvals);

			for (int n = 0; n < indicesgroups[g].size(); n++)
				d_Binarize(d_mask + Elements2(dimsimage) * n, d_mask + Elements2(dimsimage) * n, Elements2(dimsimage), h_maxvals[n] / 4.0f - 0.0f);

			//d_RectangleMask(d_mask, d_mask, toInt3(dimsimage), toInt3(dimsimage.x * 3 / 4, dimsimage.y * 3 / 4, 1), NULL, indicesgroups[g].size());

			d_WriteMRC(d_maskvolume, dimsvolume[g], "d_maskvolume" + to_string(g) + ".mrc");
			cudaFree(d_maskvolume);
			d_WriteMRC(d_mask, toInt3(dimsimage.x, dimsimage.y, indicesgroups[g].size()), "d_mask" + to_string(g) + ".mrc");
		}

		v_masks.push_back(d_mask);
	}

	for (int i = 0; i < nimages; i++)
	{
		vec(i * 5 + 0) = ToDeg(vec(i * 5 + 0)) * 10.0;
		vec(i * 5 + 1) = ToDeg(vec(i * 5 + 1));
		vec(i * 5 + 2) = ToDeg(vec(i * 5 + 2));

		constraintlower(i * 5 + 0) = ToDeg(constraintlower(i * 5 + 0)) * 10.0;
		constraintlower(i * 5 + 1) = ToDeg(constraintlower(i * 5 + 1));
		constraintlower(i * 5 + 2) = ToDeg(constraintlower(i * 5 + 2));

		constraintupper(i * 5 + 0) = ToDeg(constraintupper(i * 5 + 0)) * 10.0;
		constraintupper(i * 5 + 1) = ToDeg(constraintupper(i * 5 + 1));
		constraintupper(i * 5 + 2) = ToDeg(constraintupper(i * 5 + 2));
	}

	srand(123);
	double improvement = 9999.0;
	double freqlow = 2.0, freqhigh = dimsimage.x / 8;
	double psi = 4.0;

	TomoParamsWBPOptimizer optimizer(d_images, dimsimage, dimsvolume, nimages, indicesgroups, freqlow, freqhigh, constraintlower, constraintupper, v_masks);
	double initialscore = optimizer.Evaluate(vec);
	for (int b = 0; b < 2; b++)
	{
		improvement = 0;
		optimizer.psi = psi;
		double bestscore = optimizer.Evaluate(vec);
		column_vector bestvec = column_vector(vec);

		for (int r = 0; r < 3; r++)
		{
			vec = column_vector(bestvec);
			// Randomize everyone within psi if it isn't the first try for this level
			if (r > 0)
				for (int i = 0; i < nimages * 5; i++)
					if (std::find(indiceshold.begin(), indiceshold.end(), i / 5) == indiceshold.end())	// not in indiceshold
						vec(i) = min(constraintupper(i), max(bestvec(i) + ((double)rand() / (double)RAND_MAX - 0.5f) * psi * 2.0, constraintlower(i)));

			dlib::find_min_box_constrained(dlib::bfgs_search_strategy(),
				dlib::objective_delta_stop_strategy(1e-4),
				EvalWrapper(&optimizer),
				DerivativeWrapper(&optimizer),
				vec,
				constraintlower, constraintupper);

			// Is someone stuck?
			optimizer.Evaluate(vec);
			vector<int> stuck;
			for (int n = 0; n < nimages; n++)
				if ((optimizer.scoremean - optimizer.h_scores[n]) / optimizer.scorestd > 1.0)
					if (std::find(indiceshold.begin(), indiceshold.end(), n) == indiceshold.end())
						stuck.push_back(n);

			int stuckiteration = 0;
			vector<int> impossible;
			while (stuck.size() > 0 && stuckiteration++ < 10)
			{
				column_vector previousvec = column_vector(vec);
				double previousscore = optimizer.Evaluate(previousvec);

				// Move stuck images a bit
				int n = stuck[stuck.size() - 1];
				stuck.pop_back();
				/*for (int c = 0; c < 5; c++)
					vec(n * 5 + c) = min(constraintupper(n * 5 + c), max(previousvec(n * 5 + c) + ((double)rand() / (double)RAND_MAX - 0.5f) * psi * 4.0, constraintlower(n * 5 + c)));*/

				// Perform local stochastic search for stuck image
				double beststochasticscore = previousscore;
				column_vector beststochasticvec = column_vector(previousvec);
				// Set limits for stochastic search
				vector<tfloat2> stochasticlimits;
				stochasticlimits.push_back(tfloat2(previousvec(n * 5), previousvec(n * 5)));
				for (uint p = 1; p < 3; p++)
					stochasticlimits.push_back(tfloat2(max(constraintlower(n * 5 + p), previousvec(n * 5 + p) - psi * 2.0),
													   min(constraintupper(n * 5 + p), previousvec(n * 5 + p) + psi * 2.0)));
				for (uint p = 3; p < 5; p++)
					stochasticlimits.push_back(tfloat2(max(constraintlower(n * 5 + p), previousvec(n * 5 + p) - psi * 8.0),
													   min(constraintupper(n * 5 + p), previousvec(n * 5 + p) + psi * 8.0)));
				// Set up constraints so that only the stuck image can move
				column_vector locallower = column_vector(previousvec);
				column_vector localupper = column_vector(previousvec) + 1e-5;
				for (uint p = 0; p < 5; p++)
				{
					locallower(n * 5 + p) = constraintlower(n * 5 + p);
					localupper(n * 5 + p) = constraintupper(n * 5 + p);
				}
				optimizer.constraintlower = locallower;
				optimizer.constraintupper = localupper;

				for (uint t = 0; t < 20; t++)
				{
					// Pick random params for stuck image within specified extent
					column_vector stochasticvec = column_vector(previousvec);
					for (uint p = 0; p < 5; p++)
						stochasticvec(n * 5 + p) = (double)rand() / (double)RAND_MAX * (stochasticlimits[p].y - stochasticlimits[p].x) + stochasticlimits[p].x;

					// Starting from the random params, optimize only the stuck image
					dlib::find_min_box_constrained(dlib::bfgs_search_strategy(),
						dlib::objective_delta_stop_strategy(1e-2),
						EvalWrapper(&optimizer),
						DerivativeWrapper(&optimizer),
						stochasticvec,
						locallower, localupper);
					double stochasticscore = optimizer.Evaluate(stochasticvec);
					if (stochasticscore < beststochasticscore)
					{
						beststochasticscore = stochasticscore;
						beststochasticvec = stochasticvec;
					}
				}
				optimizer.constraintlower = constraintlower;
				optimizer.constraintupper = constraintupper;

				// Did stochastic search yield better results?
				if (previousscore - beststochasticscore > 0.1)
					vec = beststochasticvec;
				else
				{
					impossible.push_back(n);
					continue;
				}

				// Now optimize everyone with better params for stuck image
				dlib::find_min_box_constrained(dlib::bfgs_search_strategy(),
					dlib::objective_delta_stop_strategy(1e-4),
					EvalWrapper(&optimizer),
					DerivativeWrapper(&optimizer),
					vec,
					constraintlower, constraintupper);
				
				// Check if there are still some stuck images
				optimizer.Evaluate(vec);
				stuck.clear();
				for (int n = 0; n < nimages; n++)
					if ((optimizer.scoremean - optimizer.h_scores[n]) / optimizer.scorestd > 1.0)
						// Neither in indiceshold, nor in impossible
						if (std::find(indiceshold.begin(), indiceshold.end(), n) == indiceshold.end() && std::find(impossible.begin(), impossible.end(), n) == impossible.end())
							stuck.push_back(n);
			}

			// Check if score got better overall
			double currentscore = optimizer.Evaluate(vec);
			if (bestscore > currentscore)
			{
				bestvec = column_vector(vec);
				bestscore = currentscore;
			}
		}

		vec = column_vector(bestvec);
		freqlow *= 2.0;
		freqlow = min(freqlow, 4.0);
		freqhigh = min(dimsimage.x / 2.0, freqhigh * 2);
		psi *= 0.5;
		psi = max(1.0, psi);

		optimizer.freqlow = freqlow;
		optimizer.freqhigh = freqhigh;
		optimizer.Reinitialize();

		if (b == 2)
			cout << "bla";
	}

	// Store results
	finalscore = optimizer.Evaluate(vec);
	for (int n = 0; n < nimages; n++)
	{
		h_angles[n] = tfloat3(ToRad(vec(n * 5)) * 0.1f, ToRad(vec(n * 5 + 1)), ToRad(vec(n * 5 + 2)));
		h_shifts[n] = tfloat2(vec(n * 5 + 3), vec(n * 5 + 4));
	}

	for (uint g = 0; g < ngroups; g++)
		cudaFree(v_masks[g]);
}