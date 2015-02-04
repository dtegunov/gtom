#include "Prerequisites.cuh"
#include "CTF.cuh"
#include "FFT.cuh"
#include "Generics.cuh"
#include "Helper.cuh"
#include "ImageManipulation.cuh"
#include "Masking.cuh"
#include "Optimization.cuh"
#include "Transformation.cuh"

void GetPlaneScore(tfloat* d_targets, float2* d_targetcoords, uint targetlength, int2 tiles, tfloat2 spacingangstrom, int refinements, CTFFitParams fp, CTFTiltParams &tp, tfloat &score, CTFParams &bestcenterfit);

////////////////////////////////////////////////////////
//Fit specified parameters of a CTF in tilted specimen//
////////////////////////////////////////////////////////

void d_CTFTiltFit(tfloat* d_image, int2 dimsimage, CTFFitParams p, int refinements, int tilespacing, CTFTiltParams &fit, tfloat &score, tfloat &scorestddev)
{
	int2 tilesdim = toInt2(dimsimage.x / tilespacing, 
						   dimsimage.y / tilespacing);
	tfloat2 spacingangstrom = tfloat2((tfloat)tilespacing * p.pixelsize.x * 1e10, (tfloat)tilespacing * p.pixelsize.x * 1e10);
	vector<pair<tfloat3, CTFParams>> v_tiles;

	// Fit spectrum background based on the average of the entire image
	tfloat* d_background;
	uint denselength;
	{
		tfloat* d_ps;
		cudaMalloc((void**)&d_ps, ElementsFFT2(p.dimsperiodogram) * sizeof(tfloat));
		d_Periodogram(d_image, dimsimage, NULL, 0, p.dimsperiodogram, d_ps);
		d_Log(d_ps, d_ps, ElementsFFT2(p.dimsperiodogram));

		int2 dimspolar = GetCart2PolarFFTSize(p.dimsperiodogram);
		tfloat* d_pspolar;
		cudaMalloc((void**)&d_pspolar, Elements2(dimspolar) * sizeof(tfloat));
		d_Cart2PolarFFT(d_ps, d_pspolar, p.dimsperiodogram, T_INTERP_CUBIC);

		int2 dimsps = toInt2(p.maskouterradius - p.maskinnerradius, dimspolar.y);
		denselength = GetCart2PolarFFTNonredundantSize(p.dimsperiodogram, p.maskinnerradius, p.maskouterradius);
		for (int y = 0; y < dimsps.y; y++)
			cudaMemcpy(d_ps + dimsps.x * y, d_pspolar + dimspolar.x * y + p.maskinnerradius, dimsps.x * sizeof(tfloat), cudaMemcpyDeviceToDevice);
		cudaFree(d_pspolar);

		cudaMalloc((void**)&d_background, Elements2(dimsps) * sizeof(tfloat));
		d_CTFDecay(d_ps, d_background, dimsps, 4, 16);

		cudaFree(d_ps);
	}

	// Create targets using previously fitted background and denoise them with PCA
	tfloat* d_targets;
	cudaMalloc((void**)&d_targets, denselength * Elements2(tilesdim) * sizeof(tfloat));
	float2* d_targetcoords;
	cudaMalloc((void**)&d_targetcoords, denselength * sizeof(float2));
	{
		tfloat* d_tile;
		cudaMalloc((void**)&d_tile, Elements2(p.dimsperiodogram) * sizeof(tfloat));
		int3* d_origin;
		cudaMalloc((void**)&d_origin, sizeof(int3));
		int2 dimscovered = toInt2(tilesdim.x * tilespacing, tilesdim.y * tilespacing);

		for (int y = 0; y < tilesdim.y; y++)
		{
			for (int x = 0; x < tilesdim.x; x++)
			{
				int3 origin = toInt3(x * tilespacing + (dimsimage.x - dimscovered.x) / 2, y * tilespacing + (dimsimage.y - dimscovered.y) / 2, 0);
				cudaMemcpy(d_origin, &origin, sizeof(int3), cudaMemcpyHostToDevice);
				tfloat2 flatcoords = tfloat2((tfloat)(origin.x - dimsimage.x / 2), (tfloat)(origin.y - dimsimage.y / 2));

				d_ExtractMany(d_image, d_tile, toInt3(dimsimage), toInt3(p.dimsperiodogram), d_origin, 1);
				d_CTFFitCreateTarget(d_tile, p.dimsperiodogram, NULL, NULL, 0, p, d_targets + (y * tilesdim.x + x) * denselength, d_targetcoords);
			}
		}

		d_NormMonolithic(d_targets, d_targets, denselength, T_NORM_MEAN01STD, Elements2(tilesdim));

		CudaWriteToBinaryFile("d_targets.bin", d_targets, Elements2(tilesdim) * denselength * sizeof(tfloat));

		cudaFree(d_tile);
		cudaFree(d_origin);
		cudaFree(d_background);
	}

	tfloat* h_tiltscores = (tfloat*)malloc(10 * 61 * sizeof(tfloat));
	for (int phi = 0; phi <= 9; phi++)
		for (int theta = 0; theta <= 60; theta++)
		{
			fit.angles.x = ToRad((tfloat)(phi * 10));
			fit.angles.y = ToRad((tfloat)(theta - 30));
			tfloat score = 0;
			CTFParams bestparams;
			GetPlaneScore(d_targets, d_targetcoords, denselength, tilesdim, spacingangstrom, refinements, p, fit, score, bestparams);
			h_tiltscores[phi * 61 + theta] = score;
		}
	WriteToBinaryFile("d_tiltscores.bin", h_tiltscores, 10 * 61 * sizeof(tfloat));

	cudaFree(d_targetcoords);
	cudaFree(d_targets);
}

void GetPlaneScore(tfloat* d_targets, float2* d_targetcoords, uint targetlength, int2 tiles, tfloat2 spacingangstrom, int refinements, CTFFitParams fp, CTFTiltParams &tp, tfloat &bestscore, CTFParams &bestcenterfit)
{
	tfloat* d_simulated;
	cudaMalloc((void**)&d_simulated, targetlength * Elements2(tiles) * sizeof(tfloat));
	tfloat* d_score = CudaMallocValueFilled(1, (tfloat)0);

	vector<pair<tfloat, CTFParams>> v_params;
	AddCTFParamsRange(v_params, fp);

	for (int i = 0; i < refinements; i++)
	{
		for (int p = 0; p < v_params.size(); p++)
		{
			tp.centerparams = v_params[p].second;
			CTFParams* h_paramsgrid = tp.GetParamsGrid2D(tiles, spacingangstrom, tfloat3());
			d_CTFSimulate(h_paramsgrid, d_targetcoords, d_simulated, targetlength, true, Elements2(tiles));
			free(h_paramsgrid);
			//CudaWriteToBinaryFile("d_simulated.bin", d_simulated, targetlength * Elements2(tiles) * sizeof(tfloat));
			d_NormMonolithic(d_simulated, d_simulated, targetlength, T_NORM_MEAN01STD, Elements2(tiles));
			d_MultiplyByVector(d_targets, d_simulated, d_simulated, targetlength * Elements2(tiles));
			d_Sum(d_simulated, d_score, targetlength * Elements2(tiles));

			tfloat score = 0;
			cudaMemcpy(&score, d_score, sizeof(tfloat), cudaMemcpyDeviceToHost);

			score /= (tfloat)(targetlength * Elements2(tiles));
			v_params[p].first = score;
		}

		// Sort v_params by score in descending order
		sort(v_params.begin(), v_params.end(),
			[](const pair<tfloat, CTFFitParams> &a, const pair<tfloat, CTFFitParams> &b) -> bool
		{
			return a.first > b.first;
		});

		bestscore = v_params[0].first;
		bestcenterfit = v_params[0].second;

		// Decrease search step size
		tfloat3* h_fp = (tfloat3*)&fp;
		for (int j = 0; j < 11; j++)
			if (h_fp[j].x != h_fp[j].y)
				h_fp[j].z /= 4.0;

		vector<pair<tfloat, CTFParams>> v_paramsNew;
		for (int i = 0; i < min(10, (int)v_params.size()); i++)
		{
			CTFParams fit = v_params[i].second;
			CTFFitParams pNew = fp;
			tfloat3* h_p = (tfloat3*)&pNew;
			tfloat* h_f = (tfloat*)&fit;
			for (int j = 0; j < 11; j++)
				if (h_p[j].x != h_p[j].y)
					h_p[j] = tfloat3(h_f[j] - h_p[j].z * 3.0, h_f[j] + h_p[j].z * 3.0, h_p[j].z);
			AddCTFParamsRange(v_paramsNew, pNew);
		}

		v_params = v_paramsNew;
	}

	cudaFree(d_score);
	cudaFree(d_simulated);
}