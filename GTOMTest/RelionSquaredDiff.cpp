#include "Prerequisites.h"

TEST(Relion, SquaredDiff)
{
	cudaDeviceReset();

	//Case 1:
	{
		srand(123);
		int3 dims = toInt3(64, 64, 1);
		uint nrefs = 128;
		uint nshifts = 128;

		tcomplex* h_particleft = (tcomplex*)malloc(ElementsFFT(dims) * sizeof(tcomplex));
		for (int i = 0; i < ElementsFFT(dims); i++)
			h_particleft[i] = make_cuComplex(1, 1);

		tcomplex* h_precalcshiftft = (tcomplex*)malloc(ElementsFFT(dims) * nshifts * sizeof(tcomplex));
		for (uint s = 0; s < nshifts; s++)
			for (uint i = 0; i < ElementsFFT(dims); i++)
				h_precalcshiftft[s * ElementsFFT(dims) + i] = make_cuComplex(i % (s + 1) / 2, 1);

		tcomplex* h_refft = (tcomplex*)malloc(ElementsFFT(dims) * nrefs * sizeof(tcomplex));
		for (uint s = 0; s < nrefs; s++)
			for (uint i = 0; i < ElementsFFT(dims); i++)
				h_refft[s * ElementsFFT(dims) + i] = make_cuComplex(2, i % (s + 1) / 2);
		
		tcomplex* d_particleft = (tcomplex*)CudaMallocFromHostArray(h_particleft, ElementsFFT(dims) * sizeof(tcomplex));
		tfloat* d_invsigma = CudaMallocValueFilled(ElementsFFT(dims), (tfloat)1);
		tfloat* d_ctf = CudaMallocValueFilled(ElementsFFT(dims), (tfloat)1);
		tcomplex* d_refft = (tcomplex*)CudaMallocFromHostArray(h_refft, ElementsFFT(dims) * nrefs * sizeof(tcomplex));
		tcomplex* d_precalcshiftft = (tcomplex*)CudaMallocFromHostArray(h_precalcshiftft, ElementsFFT(dims) * nshifts * sizeof(tcomplex));

		uint tile = 4;
		int3* h_combination = (int3*)malloc(nshifts / tile * nrefs * sizeof(int3));
		for (uint ref = 0, i = 0; ref < nrefs; ref += tile)
			for (uint shift = 0; shift < nshifts; shift += tile, i++)
				h_combination[i] = toInt3(shift, 0, ref);
		int3* d_combination = (int3*)CudaMallocFromHostArray(h_combination, nshifts * nrefs / (tile * tile) * sizeof(int3));
		tfloat* d_diff2 = CudaMallocValueFilled(nrefs * nshifts, (tfloat)0);
		
		tfloat* h_result = MallocValueFilled(nrefs * nshifts, (tfloat)0);
		/*for (uint r = 0; r < nrefs; r++)
		{
			for (uint s = 0; s < nshifts; s++)
			{
				for (int i = 0; i < ElementsFFT(dims); i++)
				{
					tcomplex particleshifted = cmul(h_particleft[i], h_precalcshiftft[s * ElementsFFT(dims) + i]);
					tcomplex diff = make_cuComplex(h_refft[r * ElementsFFT(dims) + i].x - particleshifted.x, h_refft[r * ElementsFFT(dims) + i].y - particleshifted.y);
					h_result[r * nshifts + s] += dotp2(diff, diff);
				}
				h_result[r * nshifts + s] *= 0.5;
			}
		}*/
		for (uint b = 0; b < nshifts * nrefs / (tile * tile); b++)
			for (uint r = 0; r < tile; r++)
			{
				for (uint s = 0; s < tile; s++)
				{
					for (int i = 0; i < ElementsFFT(dims); i++)
					{
						tcomplex particleshifted = cmul(h_particleft[i], h_precalcshiftft[(h_combination[b].x + s) * ElementsFFT(dims) + i]);
						tcomplex diff = make_cuComplex(h_refft[(h_combination[b].z + r) * ElementsFFT(dims) + i].x - particleshifted.x, h_refft[(h_combination[b].z + r) * ElementsFFT(dims) + i].y - particleshifted.y);
						h_result[b * tile * tile + r * tile + s] += dotp2(diff, diff);
					}
					h_result[b * tile * tile + r * tile + s] *= 0.5;
				}
			}

		d_rlnSquaredDifferencesSparse(d_particleft, d_invsigma, d_ctf, dims, d_precalcshiftft, d_refft, d_diff2, d_combination, nshifts * nrefs, tile, false);
		//d_rlnSquaredDifferences(d_particleft, d_invsigma, d_ctf, dims, 1, d_precalcshiftft, nshifts, d_refft, nrefs, 4, d_diff2, false);

		tfloat* h_diff2 = (tfloat*)MallocFromDeviceArray(d_diff2, nrefs * nshifts * sizeof(tfloat));
		std::cout << h_diff2[0] - h_result[0];

		double sumref = 0, sumresult = 0;
		for (uint i = 0; i < nrefs * nshifts; i++)
		{
			sumref += h_result[i];
			sumresult += h_diff2[i];
		}

		std::cout << sumref << sumresult;
		assert(abs(sumref - sumresult) < 0.5);
	}

	cudaDeviceReset();
}
