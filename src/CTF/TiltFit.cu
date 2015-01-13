#include "Prerequisites.cuh"
#include "CTF.cuh"
#include "FFT.cuh"
#include "Generics.cuh"
#include "Helper.cuh"
#include "ImageManipulation.cuh"
#include "Masking.cuh"
#include "Optimization.cuh"
#include "Transformation.cuh"

void AddParamsRange(vector<pair<tfloat, CTFParams>> &v_params, CTFFitParams p);

/////////////////////////////////////
//Fit specified parameters of a CTF//
/////////////////////////////////////

void d_CTFTiltFit(tfloat* d_image, int2 dimsimage, CTFFitParams p, int refinements, int tilespacing, CTFTiltParams &fit, tfloat &score, tfloat &scorestddev)
{
	int2 tilesdim = toInt2((dimsimage.x - p.dimsperiodogram.x) / tilespacing, 
						   (dimsimage.y - p.dimsperiodogram.y) / tilespacing);
	vector<pair<tfloat3, CTFParams>> v_tiles;
	
	// Make an initial estimate for each tile using provided fit params.
	{
		tfloat* d_tile;
		cudaMalloc((void**)&d_tile, Elements2(p.dimsperiodogram) * sizeof(tfloat));
		int3* d_origin;
		cudaMalloc((void**)&d_origin, sizeof(int3));

		for (int y = 0; y < tilesdim.y; y++)
		{
			for (int x = 0; x < tilesdim.x; x++)
			{
				int3 origin = toInt3(x * tilespacing, y * tilespacing, 0);
				cudaMemcpy(d_origin, &origin, sizeof(int3), cudaMemcpyHostToDevice);
				tfloat2 flatcoords = tfloat2((tfloat)(origin.x - dimsimage.x / 2), (tfloat)(origin.y - dimsimage.y / 2));

				d_Extract(d_image, d_tile, toInt3(dimsimage), toInt3(p.dimsperiodogram), d_origin, 1);

				CTFParams tileparams;
				tfloat tilescore = 0, mean = 0, stddev = 0;
				d_CTFFit(d_tile, p.dimsperiodogram, NULL, 0, p, refinements, tileparams, tilescore, mean, stddev);

				v_tiles.push_back(pair<tfloat3, CTFParams>(tfloat3(flatcoords.x, flatcoords.y, tilescore), tileparams));
			}
		}

		cudaFree(d_tile);
		cudaFree(d_origin);
	}

	// Estimate mean parameters based on the best 10 % of all fits,
	// and the defocus value at the center based on all fits.
	CTFParams centerparams;
	{
		// First estimate for mean defocus based on all fits.
		double meandefocus = 0;
		for (int i = 0; i < Elements2(tilesdim); i++)
			meandefocus += v_tiles[i].second.defocus;
		meandefocus /= (double)Elements2(tilesdim);

		// Sort tiles by fit CC in descending order.
		sort(v_tiles.begin(), v_tiles.end(),
			[](const pair<tfloat3, CTFFitParams> &a, const pair<tfloat3, CTFFitParams> &b) -> bool
		{
			return a.first.z > b.first.z;
		});

		double* h_mean = MallocValueFilled(11, 0.0);
		int bestfraction = (double)v_tiles.size() * 0.1;

		#pragma omp for
		for (int j = 0; j < 11; j++)
			for (int i = 0; i < bestfraction; i++)
				h_mean[j] += ((tfloat*)&v_tiles[i].second)[j];

		for (int i = 0; i < 11; i++)
			if (((tfloat3*)&p)[i].x != ((tfloat3*)&p)[i].y)	// If this param is supposed to be estimated.
				((tfloat*)&centerparams)[i] = (tfloat)(h_mean[i] / (double)bestfraction);
		centerparams.defocus = meandefocus;
	}

	WriteToBinaryFile("d_centerparams.bin", (tfloat*)&centerparams, 11 * sizeof(tfloat));

	tfloat* h_zestimate = (tfloat*)malloc(Elements2(tilesdim) * sizeof(tfloat));
	for (int i = 0; i < Elements2(tilesdim); i++)
		h_zestimate[i] = v_tiles[i].second.defocus;

	WriteToBinaryFile("d_zestimate.bin", h_zestimate, Elements2(tilesdim) * sizeof(tfloat));
}