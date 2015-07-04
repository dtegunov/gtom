#include "Prerequisites.cuh"

#define TpB 128
#define ReduceTo 8

namespace gtom
{
	template<uint tile, bool dofirstitercc> __global__ void SquaredDifferencesKernel(const tcomplex* __restrict__ d_particleft, const tfloat* __restrict__ d_minvsigma2, const tfloat* __restrict__ d_ctf, uint elements, const tcomplex* __restrict__ d_precalcshifts, const tcomplex* __restrict__ d_refft, tfloat* d_diff2);
	template<uint tile, bool dofirstitercc> __global__ void SquaredDifferences180Kernel(const tcomplex* __restrict__ d_particleft, const tfloat* __restrict__ d_minvsigma2, const tfloat* __restrict__ d_ctf, uint elements, const tcomplex* __restrict__ d_precalcshifts, const tcomplex* __restrict__ d_refft, tfloat* d_diff2, uint npsihalf, uint minref);
	template<uint tile, bool dofirstitercc> __global__ void SquaredDifferencesSparseKernel(const tcomplex* __restrict__ d_particleft, const tfloat* __restrict__ d_minvsigma2, const tfloat* __restrict__ d_ctf, uint elements, const tcomplex* __restrict__ d_precalcshifts, const tcomplex* __restrict__ d_refft, tfloat* d_diff2, const uint3* __restrict__ d_combination);

	void d_rlnSquaredDifferences(tcomplex* d_particleft, tfloat* d_minvsigma2, tfloat* d_ctf, int3 dimsparticle, uint nparticles, tcomplex* d_precalcshifts, uint nshifts, tcomplex* d_refft, uint nrefs, uint tile, tfloat* d_diff2, bool dofirstitercc)
	{
		uint elements = ElementsFFT(dimsparticle);
		dim3 grid = dim3((nshifts + tile - 1) / tile, nparticles, (nrefs + tile - 1) / tile);
		if (tile == 1)
		{
			if (dofirstitercc)
				SquaredDifferencesKernel<1, true> << <grid, TpB >> >(d_particleft, d_minvsigma2, d_ctf, elements, d_precalcshifts, d_refft, d_diff2);
			else
				SquaredDifferencesKernel<1, false> << <grid, TpB >> >(d_particleft, d_minvsigma2, d_ctf, elements, d_precalcshifts, d_refft, d_diff2);
		}
		else if (tile == 2)
		{
			if (dofirstitercc)
				SquaredDifferencesKernel<2, true> << <grid, TpB >> >(d_particleft, d_minvsigma2, d_ctf, elements, d_precalcshifts, d_refft, d_diff2);
			else
				SquaredDifferencesKernel<2, false> << <grid, TpB >> >(d_particleft, d_minvsigma2, d_ctf, elements, d_precalcshifts, d_refft, d_diff2);
		}
		else if (tile == 3)
		{
			if (dofirstitercc)
				SquaredDifferencesKernel<3, true> << <grid, TpB >> >(d_particleft, d_minvsigma2, d_ctf, elements, d_precalcshifts, d_refft, d_diff2);
			else
				SquaredDifferencesKernel<3, false> << <grid, TpB >> >(d_particleft, d_minvsigma2, d_ctf, elements, d_precalcshifts, d_refft, d_diff2);
		}
		else if (tile == 4)
		{
			if (dofirstitercc)
				SquaredDifferencesKernel<4, true> << <grid, TpB >> >(d_particleft, d_minvsigma2, d_ctf, elements, d_precalcshifts, d_refft, d_diff2);
			else
				SquaredDifferencesKernel<4, false> << <grid, TpB >> >(d_particleft, d_minvsigma2, d_ctf, elements, d_precalcshifts, d_refft, d_diff2);
		}
		else
			throw;
	}

	void d_rlnSquaredDifferences180(tcomplex* d_particleft, tfloat* d_minvsigma2, tfloat* d_ctf, int3 dimsparticle, uint nparticles, tcomplex* d_precalcshifts, uint nshifts, tcomplex* d_refft, uint nrefs, uint npsi, uint minref, uint tile, tfloat* d_diff2, bool dofirstitercc)
	{
		uint elements = ElementsFFT(dimsparticle);
		dim3 grid = dim3((nshifts + tile - 1) / tile, nparticles, (nrefs + tile - 1) / tile);
		if (tile == 1)
		{
			if (dofirstitercc)
				SquaredDifferences180Kernel<1, true> << <grid, TpB >> >(d_particleft, d_minvsigma2, d_ctf, elements, d_precalcshifts, d_refft, d_diff2, npsi / 2, minref);
			else
				SquaredDifferences180Kernel<1, false> << <grid, TpB >> >(d_particleft, d_minvsigma2, d_ctf, elements, d_precalcshifts, d_refft, d_diff2, npsi / 2, minref);
		}
		else if (tile == 2)
		{
			if (dofirstitercc)
				SquaredDifferences180Kernel<2, true> << <grid, TpB >> >(d_particleft, d_minvsigma2, d_ctf, elements, d_precalcshifts, d_refft, d_diff2, npsi / 2, minref);
			else
				SquaredDifferences180Kernel<2, false> << <grid, TpB >> >(d_particleft, d_minvsigma2, d_ctf, elements, d_precalcshifts, d_refft, d_diff2, npsi / 2, minref);
		}
		else if (tile == 3)
		{
			if (dofirstitercc)
				SquaredDifferences180Kernel<3, true> << <grid, TpB >> >(d_particleft, d_minvsigma2, d_ctf, elements, d_precalcshifts, d_refft, d_diff2, npsi / 2, minref);
			else
				SquaredDifferences180Kernel<3, false> << <grid, TpB >> >(d_particleft, d_minvsigma2, d_ctf, elements, d_precalcshifts, d_refft, d_diff2, npsi / 2, minref);
		}
		else if (tile == 4)
		{
			if (dofirstitercc)
				SquaredDifferences180Kernel<4, true> << <grid, TpB >> >(d_particleft, d_minvsigma2, d_ctf, elements, d_precalcshifts, d_refft, d_diff2, npsi / 2, minref);
			else
				SquaredDifferences180Kernel<4, false> << <grid, TpB >> >(d_particleft, d_minvsigma2, d_ctf, elements, d_precalcshifts, d_refft, d_diff2, npsi / 2, minref);
		}
		else
			throw;
	}

	void d_rlnSquaredDifferencesSparse(tcomplex* d_particleft, tfloat* d_minvsigma2, tfloat* d_ctf, int3 dimsparticle, tcomplex* d_precalcshifts, tcomplex* d_refft, tfloat* d_diff2, uint3* d_combination, uint ncombinations, uint tile, bool dofirstitercc)
	{
		uint elements = ElementsFFT(dimsparticle);
		dim3 grid = dim3(ncombinations / (tile * tile));
		
		if (tile == 1)
		{
			if (dofirstitercc)
				SquaredDifferencesSparseKernel<1, true> << <grid, TpB >> >(d_particleft, d_minvsigma2, d_ctf, elements, d_precalcshifts, d_refft, d_diff2, d_combination);
			else
				SquaredDifferencesSparseKernel<1, false> << <grid, TpB >> >(d_particleft, d_minvsigma2, d_ctf, elements, d_precalcshifts, d_refft, d_diff2, d_combination);
		}
		else if (tile == 2)
		{
			if (dofirstitercc)
				SquaredDifferencesSparseKernel<2, true> << <grid, TpB >> >(d_particleft, d_minvsigma2, d_ctf, elements, d_precalcshifts, d_refft, d_diff2, d_combination);
			else
				SquaredDifferencesSparseKernel<2, false> << <grid, TpB >> >(d_particleft, d_minvsigma2, d_ctf, elements, d_precalcshifts, d_refft, d_diff2, d_combination);
		}
		else if (tile == 3)
		{
			if (dofirstitercc)
				SquaredDifferencesSparseKernel<3, true> << <grid, TpB >> >(d_particleft, d_minvsigma2, d_ctf, elements, d_precalcshifts, d_refft, d_diff2, d_combination);
			else
				SquaredDifferencesSparseKernel<3, false> << <grid, TpB >> >(d_particleft, d_minvsigma2, d_ctf, elements, d_precalcshifts, d_refft, d_diff2, d_combination);
		}
		else if (tile == 4)
		{
			if (dofirstitercc)
				SquaredDifferencesSparseKernel<4, true> << <grid, TpB >> >(d_particleft, d_minvsigma2, d_ctf, elements, d_precalcshifts, d_refft, d_diff2, d_combination);
			else
				SquaredDifferencesSparseKernel<4, false> << <grid, TpB >> >(d_particleft, d_minvsigma2, d_ctf, elements, d_precalcshifts, d_refft, d_diff2, d_combination);
		}
		else
			throw;
	}

	template<uint tile, bool dofirstitercc> __global__ void SquaredDifferencesKernel(const tcomplex* __restrict__ d_particleft, 
																					 const tfloat* __restrict__ d_minvsigma2, 
																					 const tfloat* __restrict__ d_ctf, 
																					 uint elements, 
																					 const tcomplex* __restrict__ d_precalcshifts, 
																					 const tcomplex* __restrict__ d_refft, 
																					 tfloat* d_diff2)
	{
		__shared__ tfloat ss_diff2[(TpB / 32) * tile * tile * ReduceTo];
		__shared__ tfloat ss_suma2[dofirstitercc ? (TpB / 32) * tile * tile * ReduceTo : 1];
		tfloat* s_diff2 = &ss_diff2[0];
		tfloat* s_suma2 = &ss_suma2[0];
		for (uint i = threadIdx.x; i < TpB / 32 * tile * tile * ReduceTo; i += TpB)
		{
			s_diff2[i] = 0;
			if (dofirstitercc)
				s_suma2[i] = 0;
		}
		__syncthreads();

		d_particleft += elements * blockIdx.y;
		d_precalcshifts += elements * blockIdx.x * tile;

		// Offset to current refbatch
		d_refft += elements * tile * blockIdx.z;
		d_diff2 += ((tile * blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x) * tile;

		d_minvsigma2 += elements * blockIdx.y;
		d_ctf += elements * blockIdx.y;

		d_refft += threadIdx.x;
		d_particleft += threadIdx.x;
		d_precalcshifts += threadIdx.x;
		d_minvsigma2 += threadIdx.x;
		d_ctf += threadIdx.x;

		tcomplex r_refft[tile];

		tfloat diff2, suma2;
		ushort withinwarp = threadIdx.x % 32;
		ushort warpid = threadIdx.x / 32;

		s_diff2 += warpid * ReduceTo + withinwarp;
		if (dofirstitercc)
			s_suma2 += warpid * ReduceTo + withinwarp;

		for (uint id = threadIdx.x; id < elements; id += TpB, d_particleft += TpB, d_minvsigma2 += TpB, d_ctf += TpB, d_refft += TpB, d_precalcshifts += TpB)
		{
			tcomplex particleft = *d_particleft;
			tfloat minvsigma2 = dofirstitercc ? 0 : *d_minvsigma2;
			tfloat ctf = *d_ctf;

			for (uint r = 0; r < tile; r++)
				r_refft[r] = d_refft[r * elements] * ctf;

			for (uint s = 0; s < tile; s++)
			{
				tcomplex particleshifted = cmul(particleft, d_precalcshifts[s * elements]);
				for (uint r = 0; r < tile; r++)
				{
					if (dofirstitercc)
					{
						diff2 = dotp2(particleshifted, r_refft[r]);
						suma2 = dotp2(r_refft[r], r_refft[r]);
					}
					else
					{
						tcomplex diff = r_refft[r] - particleshifted;
						diff2 = dotp2(diff, diff) * minvsigma2;
					}

					for (uint w = 16; w > ReduceTo / 2; w /= 2)
					{
						diff2 += __shfl_down(diff2, w);
						if (dofirstitercc)
							suma2 += __shfl_down(suma2, w);
					}
					if (withinwarp < ReduceTo)
					{
						s_diff2[(s * tile + r) * TpB / 32 * ReduceTo] += diff2;
						if (dofirstitercc)
							s_suma2[(s * tile + r) * TpB / 32 * ReduceTo] += suma2;
					}
				}
			}
			diff2 = 0;
			if (dofirstitercc)
				suma2 = 0;
		}

		s_diff2 -= warpid * ReduceTo + withinwarp;
		if (dofirstitercc)
			s_suma2 -= warpid * ReduceTo + withinwarp;

		__syncthreads();

		for (uint id = threadIdx.x; id < (TpB / 32) * tile * tile * ReduceTo; id += TpB)
		{
			diff2 = s_diff2[id];
			if (dofirstitercc)
				suma2 = s_suma2[id];

			for (uint w = TpB / 64 * ReduceTo; w > 0; w /= 2)
			{
				diff2 += __shfl_down(diff2, w);
				if (dofirstitercc)
					suma2 += __shfl_down(suma2, w);
			}

			if (id % (TpB / 32 * ReduceTo) == 0)
			{
				uint s = (id / (TpB / 32 * ReduceTo)) / tile;
				uint r = (id / (TpB / 32 * ReduceTo)) % tile;
				if (dofirstitercc)
					diff2 *= rsqrt(suma2);
				else
					diff2 *= 0.5f;

				d_diff2[r * gridDim.y * (gridDim.x * tile) + s] = diff2;
			}
		}
	}

	template<uint tile, bool dofirstitercc> __global__ void SquaredDifferences180Kernel(const tcomplex* __restrict__ d_particleft, 
																						const tfloat* __restrict__ d_minvsigma2, 
																						const tfloat* __restrict__ d_ctf, 
																						uint elements, 
																						const tcomplex* __restrict__ d_precalcshifts, 
																						const tcomplex* __restrict__ d_refft, 
																						tfloat* d_diff2, 
																						uint npsihalf,
																						uint minref)
	{
		__shared__ tfloat ss_diff2[(TpB / 32) * tile * tile * ReduceTo * 2];
		__shared__ tfloat ss_suma2[dofirstitercc ? (TpB / 32) * tile * tile * ReduceTo * 2 : 1];
		tfloat* s_diff2 = &ss_diff2[0];
		tfloat* s_suma2 = &ss_suma2[0];
		for (uint i = threadIdx.x; i < TpB / 32 * tile * tile * ReduceTo * 2; i += TpB)
		{
			s_diff2[i] = 0;
			if (dofirstitercc)
				s_suma2[i] = 0;
		}
		__syncthreads();

		d_particleft += elements * blockIdx.y;
		d_precalcshifts += elements * blockIdx.x * tile;

		// Offset to current refbatch
		d_diff2 += blockIdx.x * tile;
		d_refft += elements * tile * blockIdx.z;

		d_minvsigma2 += elements * blockIdx.y;
		d_ctf += elements * blockIdx.y;

		d_refft += threadIdx.x;
		d_particleft += threadIdx.x;
		d_precalcshifts += threadIdx.x;
		d_minvsigma2 += threadIdx.x;
		d_ctf += threadIdx.x;

		tcomplex r_refft[tile];

		tfloat diff2, suma2;
		ushort withinwarp = threadIdx.x % 32;
		ushort warpid = threadIdx.x / 32;

		s_diff2 += warpid * ReduceTo + withinwarp;
		if (dofirstitercc)
			s_suma2 += warpid * ReduceTo + withinwarp;

		for (uint id = threadIdx.x; id < elements; id += TpB, d_particleft += TpB, d_minvsigma2 += TpB, d_ctf += TpB, d_refft += TpB, d_precalcshifts += TpB)
		{
			tcomplex particleft = *d_particleft;
			tfloat minvsigma2 = dofirstitercc ? 0 : *d_minvsigma2;
			tfloat ctf = *d_ctf;

			for (uint r = 0; r < tile; r++)
				r_refft[r] = d_refft[r * elements] * ctf;

			for (uint s = 0; s < tile; s++)
			{
				tcomplex particleshifted = cmul(particleft, d_precalcshifts[s * elements]);
				for (uint r = 0; r < tile; r++)
				{
					if (dofirstitercc)
					{
						diff2 = dotp2(particleshifted, r_refft[r]);
						suma2 = dotp2(r_refft[r], r_refft[r]);
					}
					else
					{
						tcomplex diff = r_refft[r] - particleshifted;
						diff2 = dotp2(diff, diff) * minvsigma2;
					}

					for (uint w = 16; w > ReduceTo / 2; w /= 2)
					{
						diff2 += __shfl_down(diff2, w);
						if (dofirstitercc)
							suma2 += __shfl_down(suma2, w);
					}
					if (withinwarp < ReduceTo)
					{
						s_diff2[(s * tile + r) * TpB / 32 * ReduceTo] += diff2;
						if (dofirstitercc)
							s_suma2[(s * tile + r) * TpB / 32 * ReduceTo] += suma2;
					}

					// Take conjugate ref value, i. e. perform a 180 deg rotation
					if (dofirstitercc)
					{
						diff2 = dotp2(particleshifted, cconj(r_refft[r]));
						suma2 = dotp2(cconj(r_refft[r]), cconj(r_refft[r]));
					}
					else
					{
						tcomplex diff = cconj(r_refft[r]) - particleshifted;
						diff2 = dotp2(diff, diff) * minvsigma2;
					}

					for (uint w = 16; w > ReduceTo / 2; w /= 2)
					{
						diff2 += __shfl_down(diff2, w);
						if (dofirstitercc)
							suma2 += __shfl_down(suma2, w);
					}
					if (withinwarp < ReduceTo)
					{
						s_diff2[(s * tile + r) * TpB / 32 * ReduceTo + tile * tile * TpB / 32 * ReduceTo] += diff2;
						if (dofirstitercc)
							s_suma2[(s * tile + r) * TpB / 32 * ReduceTo + tile * tile * TpB / 32 * ReduceTo] += suma2;
					}
				}
			}
			diff2 = 0;
			if (dofirstitercc)
				suma2 = 0;
		}

		s_diff2 -= warpid * ReduceTo + withinwarp;
		if (dofirstitercc)
			s_suma2 -= warpid * ReduceTo + withinwarp;

		__syncthreads();

		uint orientation = minref + blockIdx.z * tile;
		uint sigmatheta = orientation / npsihalf;
		uint psi = orientation - sigmatheta * npsihalf;

		for (uint id = threadIdx.x; id < (TpB / 32) * tile * tile * ReduceTo; id += TpB)
		{
			diff2 = s_diff2[id];
			if (dofirstitercc)
				suma2 = s_suma2[id];

			for (uint w = TpB / 64 * ReduceTo; w > 0; w /= 2)
			{
				diff2 += __shfl_down(diff2, w);
				if (dofirstitercc)
					suma2 += __shfl_down(suma2, w);
			}

			if (id % (TpB / 32 * ReduceTo) == 0)
			{
				uint s = (id / (TpB / 32 * ReduceTo)) / tile;
				uint r = (id / (TpB / 32 * ReduceTo)) % tile;
				if (dofirstitercc)
					diff2 *= rsqrt(suma2);
				else
					diff2 *= 0.5f;

				d_diff2[((sigmatheta * npsihalf * 2 + psi + r) * gridDim.y + blockIdx.y) * (gridDim.x * tile) + s] = diff2;
			}
		}

		s_diff2 += tile * tile * TpB / 32 * ReduceTo;
		if (dofirstitercc)
			s_suma2 += tile * tile * TpB / 32 * ReduceTo;

		// Deal with the psi + 180 deg values
		for (uint id = threadIdx.x; id < (TpB / 32) * tile * tile * ReduceTo; id += TpB)
		{
			diff2 = s_diff2[id];
			if (dofirstitercc)
				suma2 = s_suma2[id];

			for (uint w = TpB / 64 * ReduceTo; w > 0; w /= 2)
			{
				diff2 += __shfl_down(diff2, w);
				if (dofirstitercc)
					suma2 += __shfl_down(suma2, w);
			}

			if (id % (TpB / 32 * ReduceTo) == 0)
			{
				uint s = (id / (TpB / 32 * ReduceTo)) / tile;
				uint r = (id / (TpB / 32 * ReduceTo)) % tile;
				if (dofirstitercc)
					diff2 *= rsqrt(suma2);
				else
					diff2 *= 0.5f;

				d_diff2[((sigmatheta * npsihalf * 2 + psi + npsihalf + r) * gridDim.y + blockIdx.y) * (gridDim.x * tile) + s] = diff2;
			}
		}
	}

	template<uint tile, bool dofirstitercc> __global__ void SquaredDifferencesSparseKernel(const tcomplex* __restrict__ d_particleft, 
																						   const tfloat* __restrict__ d_minvsigma2, 
																						   const tfloat* __restrict__ d_ctf, 
																						   uint elements, 
																						   const tcomplex* __restrict__ d_precalcshifts, 
																						   const tcomplex* __restrict__ d_refft, 
																						   tfloat* d_diff2, 
																						   const uint3* __restrict__ d_combination)
	{
		__shared__ tfloat ss_diff2[(TpB / 32) * tile * tile * ReduceTo];
		__shared__ tfloat ss_suma2[dofirstitercc ? (TpB / 32) * tile * tile * ReduceTo : 1];
		tfloat* s_diff2 = &ss_diff2[0];
		tfloat* s_suma2 = &ss_suma2[0];
		for (uint i = threadIdx.x; i < TpB / 32 * tile * tile * ReduceTo; i += TpB)
		{
			s_diff2[i] = 0;
			if (dofirstitercc)
				s_suma2[i] = 0;
		}
		__syncthreads();

		d_particleft += elements * d_combination[blockIdx.x].y;
		d_precalcshifts += elements * d_combination[blockIdx.x].x;

		// Offset to current refbatch
		d_refft += elements * d_combination[blockIdx.x].z;
		d_diff2 += blockIdx.x * tile * tile;

		d_minvsigma2 += elements * d_combination[blockIdx.x].y;
		d_ctf += elements * d_combination[blockIdx.x].y;

		d_refft += threadIdx.x;
		d_particleft += threadIdx.x;
		d_precalcshifts += threadIdx.x;
		d_minvsigma2 += threadIdx.x;
		d_ctf += threadIdx.x;

		tcomplex r_refft[tile];

		tfloat diff2, suma2;
		ushort withinwarp = threadIdx.x % 32;
		ushort warpid = threadIdx.x / 32;

		s_diff2 += warpid * ReduceTo + withinwarp;
		if (dofirstitercc)
			s_suma2 += warpid * ReduceTo + withinwarp;

		for (uint id = threadIdx.x; id < elements; id += TpB, d_particleft += TpB, d_minvsigma2 += TpB, d_ctf += TpB, d_refft += TpB, d_precalcshifts += TpB)
		{
			tcomplex particleft = *d_particleft;
			tfloat minvsigma2 = dofirstitercc ? 0 : *d_minvsigma2;
			tfloat ctf = *d_ctf;

			for (uint r = 0; r < tile; r++)
				r_refft[r] = d_refft[r * elements] * ctf;

			for (uint s = 0; s < tile; s++)
			{
				tcomplex particleshifted = cmul(particleft, d_precalcshifts[s * elements]);
				for (uint r = 0; r < tile; r++)
				{
					if (dofirstitercc)
					{
						diff2 = dotp2(particleshifted, r_refft[r]);
						suma2 = dotp2(r_refft[r], r_refft[r]);
					}
					else
					{
						tcomplex diff = r_refft[r] - particleshifted;
						diff2 = dotp2(diff, diff) * minvsigma2;
					}

					for (uint w = 16; w > ReduceTo / 2; w /= 2)
					{
						diff2 += __shfl_down(diff2, w);
						if (dofirstitercc)
							suma2 += __shfl_down(suma2, w);
					}
					if (withinwarp < ReduceTo)
					{
						s_diff2[(s * tile + r) * TpB / 32 * ReduceTo] += diff2;
						if (dofirstitercc)
							s_suma2[(s * tile + r) * TpB / 32 * ReduceTo] += suma2;
					}
				}
			}
			diff2 = 0;
			if (dofirstitercc)
				suma2 = 0;
		}

		s_diff2 -= warpid * ReduceTo + withinwarp;
		if (dofirstitercc)
			s_suma2 -= warpid * ReduceTo + withinwarp;

		__syncthreads();

		for (uint id = threadIdx.x; id < (TpB / 32) * tile * tile * ReduceTo; id += TpB)
		{
			diff2 = s_diff2[id];
			if (dofirstitercc)
				suma2 = s_suma2[id];

			for (uint w = TpB / 64 * ReduceTo; w > 0; w /= 2)
			{
				diff2 += __shfl_down(diff2, w);
				if (dofirstitercc)
					suma2 += __shfl_down(suma2, w);
			}

			if (id % (TpB / 32 * ReduceTo) == 0)
			{
				uint s = (id / (TpB / 32 * ReduceTo)) / tile;
				uint r = (id / (TpB / 32 * ReduceTo)) % tile;
				if (dofirstitercc)
					diff2 *= rsqrt(suma2);
				else
					diff2 *= 0.5f;

				d_diff2[r * tile + s] = diff2;
			}
		}
	}
}
