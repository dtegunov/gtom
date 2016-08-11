#include "Prerequisites.cuh"
#include "Angles.cuh"
#include "Helper.cuh"
#include "Relion.cuh"

namespace gtom
{
	//template<uint TpB> __global__ void Project2Dto2DKernel(cudaTex t_volumeRe, cudaTex t_volumeIm, uint dimvolume, tcomplex* d_proj, uint dimproj, uint rmax, uint rmax2);
	template<uint ndims, uint TpB> __global__ void Backproject3DtoNDKernel(tcomplex* d_volumeft, tfloat* d_volumeweights, uint dimvolume, tcomplex* d_projft, tfloat* d_projweights, uint dimproj, size_t elementsproj, uint rmax, int rmax2);

	__constant__ float c_backmatrices[128 * 9];

	void d_rlnBackproject(tcomplex* d_volumeft, tfloat* d_volumeweights, int3 dimsvolume, tcomplex* d_projft, tfloat* d_projweights, int3 dimsproj, uint rmax, tfloat3* h_angles, float supersample, uint batch)
	{
		glm::mat3* d_matrices;

		{
			glm::mat3* h_matrices = (glm::mat3*)malloc(sizeof(glm::mat3) * batch);
			for (int i = 0; i < batch; i++)
				h_matrices[i] = glm::transpose(Matrix3Euler(h_angles[i])) * Matrix3Scale(supersample);
			d_matrices = (glm::mat3*)CudaMallocFromHostArray(h_matrices, sizeof(glm::mat3) * batch);
			free(h_matrices);
		}

		d_rlnBackproject(d_volumeft, d_volumeweights, dimsvolume, d_projft, d_projweights, dimsproj, rmax, d_matrices, batch);

		{
			cudaFree(d_matrices);
		}
	}

	void d_rlnBackproject(tcomplex* d_volumeft, tfloat* d_volumeweights, int3 dimsvolume, tcomplex* d_projft, tfloat* d_projweights, int3 dimsproj, uint rmax, glm::mat3* d_matrices, uint batch)
	{
		uint ndimsvolume = DimensionCount(dimsvolume);
		uint ndimsproj = DimensionCount(dimsproj);
		if (ndimsvolume < ndimsproj)
			throw;

		rmax = tmin(rmax, dimsproj.x / 2);

		for (uint b = 0; b < batch; b += 128)
		{
			uint curbatch = tmin(128, batch - b);

			if (ndimsvolume == 3)
			{
				cudaMemcpyToSymbol(c_backmatrices, d_matrices + b, curbatch * sizeof(glm::mat3), 0, cudaMemcpyDeviceToDevice);

				dim3 grid = dim3(1, curbatch, 1);
				uint elements = ElementsFFT(dimsproj);

				if (ndimsproj == 2)
					Backproject3DtoNDKernel<2, 128> << <grid, 128 >> > (d_volumeft, d_volumeweights, dimsvolume.x, d_projft + elements * b, d_projweights + elements * b, dimsproj.x, elements, rmax, rmax * rmax);
				else if (ndimsproj == 3)
					Backproject3DtoNDKernel<3, 128> << <grid, 128 >> > (d_volumeft, d_volumeweights, dimsvolume.x, d_projft + elements * b, d_projweights + elements * b, dimsproj.x, elements, rmax, rmax * rmax);
			}
			else
			{
				/*cudaMemcpyToSymbol(c_backmatrices, d_matrices, batch * sizeof(glm::mat3), 0, cudaMemcpyDeviceToDevice);

				dim3 grid = dim3(1, batch, 1);
				uint elements = ElementsFFT(dimsproj);
				uint TpB = 1 << tmin(7, tmax(7, (uint)(log(elements / 4.0) / log(2.0))));

				if (TpB == 32)
				Project2Dto2DKernel<32> << <grid, 32 >> > (t_volumeRe, t_volumeIm, dimsvolume.x, d_proj, dimsproj.x, rmax, rmax * rmax);
				else if (TpB == 64)
				Project2Dto2DKernel<64> << <grid, 64 >> > (t_volumeRe, t_volumeIm, dimsvolume.x, d_proj, dimsproj.x, rmax, rmax * rmax);
				else if (TpB == 128)
				Project2Dto2DKernel<128> << <grid, 128 >> > (t_volumeRe, t_volumeIm, dimsvolume.x, d_proj, dimsproj.x, rmax, rmax * rmax);
				else if (TpB == 256)
				Project2Dto2DKernel<256> << <grid, 256 >> > (t_volumeRe, t_volumeIm, dimsvolume.x, d_proj, dimsproj.x, rmax, rmax * rmax);
				else
				throw;*/
			}
		}
	}

	template<uint ndims, uint TpB> __global__ void Backproject3DtoNDKernel(tcomplex* d_volumeft, tfloat* d_volumeweights, uint dimvolume, tcomplex* d_projft, tfloat* d_projweights, uint dimproj, size_t elementsproj, uint rmax, int rmax2)
	{
		d_projft += elementsproj * blockIdx.y;
		d_projweights += elementsproj * blockIdx.y;
		
		uint slice = ndims == 3 ? ElementsFFT1(dimproj) * dimproj : 1;
		uint dimft = ElementsFFT1(dimproj);
		uint dimvolumeft = ElementsFFT1(dimvolume);

		for (uint id = threadIdx.x; id < elementsproj; id += TpB)
		{
			uint idx = id % dimft;
			uint idy = (ndims == 3 ? id % slice : id) / dimft;
			uint idz = ndims == 3 ? id / slice : 0;

			int x = idx;
			int y = idy <= rmax ? idy : (int)idy - (int)dimproj;
			int z = ndims == 3 ? (idz <= rmax ? idz : (int)idz - (int)dimproj) : 0;
			int r2 = ndims == 3 ? z * z + y * y + x * x : y * y + x * x;
			if (r2 > rmax2)
				continue;

			glm::vec3 pos = glm::vec3(x, y, z);
			pos = glm::vec3(c_backmatrices[blockIdx.y * 9 + 0] * pos.x + c_backmatrices[blockIdx.y * 9 + 3] * pos.y + c_backmatrices[blockIdx.y * 9 + 6] * pos.z,
							c_backmatrices[blockIdx.y * 9 + 1] * pos.x + c_backmatrices[blockIdx.y * 9 + 4] * pos.y + c_backmatrices[blockIdx.y * 9 + 7] * pos.z,
							c_backmatrices[blockIdx.y * 9 + 2] * pos.x + c_backmatrices[blockIdx.y * 9 + 5] * pos.y + c_backmatrices[blockIdx.y * 9 + 8] * pos.z);

			// Only asymmetric half is stored
			float is_neg_x = 1.0f;
			if (pos.x + 1e-5f < 0)
			{
				// Get complex conjugated hermitian symmetry pair
				pos.x = abs(pos.x);
				pos.y = -pos.y;
				pos.z = -pos.z;
				is_neg_x = -1.0f;
			}

			// Bilinear interpolation (with physical coords)
			// Subtract STARTINGY to accelerate access to data (STARTINGX=0)
			// In that way use DIRECT_A3D_ELEM, rather than A3D_ELEM
			int x0 = floor(pos.x + 1e-5f);
			pos.x -= x0;
			int x1 = x0 + 1;

			int y0 = floor(pos.y);
			pos.y -= y0;
			y0 += dimvolume / 2;
			int y1 = y0 + 1;

			int z0 = floor(pos.z);
			pos.z -= z0;
			z0 += dimvolume / 2;
			int z1 = z0 + 1;

			float c0 = 1.0f - pos.z;
			float c1 = pos.z;

			float c00 = (1.0f - pos.y) * c0;
			float c10 = pos.y * c0;
			float c01 = (1.0f - pos.y) * c1;
			float c11 = pos.y * c1;

			float c000 = (1.0f - pos.x) * c00;
			float c100 = pos.x * c00;
			float c010 = (1.0f - pos.x) * c10;
			float c110 = pos.x * c10;
			float c001 = (1.0f - pos.x) * c01;
			float c101 = pos.x * c01;
			float c011 = (1.0f - pos.x) * c11;
			float c111 = pos.x * c11;

			tcomplex val = d_projft[id];
			val.y *= is_neg_x;
			tfloat weight = d_projweights[id];

			atomicAdd((tfloat*)(d_volumeft + (z0 * dimvolume + y0) * dimvolumeft + x0), c000 * val.x);
			atomicAdd((tfloat*)(d_volumeft + (z0 * dimvolume + y0) * dimvolumeft + x0) + 1, c000 * val.y);
			atomicAdd((tfloat*)(d_volumeweights + (z0 * dimvolume + y0) * dimvolumeft + x0), c000 * weight);

			atomicAdd((tfloat*)(d_volumeft + (z0 * dimvolume + y0) * dimvolumeft + x1), c100 * val.x);
			atomicAdd((tfloat*)(d_volumeft + (z0 * dimvolume + y0) * dimvolumeft + x1) + 1, c100 * val.y);
			atomicAdd((tfloat*)(d_volumeweights + (z0 * dimvolume + y0) * dimvolumeft + x1), c100 * weight);

			atomicAdd((tfloat*)(d_volumeft + (z0 * dimvolume + y1) * dimvolumeft + x0), c010 * val.x);
			atomicAdd((tfloat*)(d_volumeft + (z0 * dimvolume + y1) * dimvolumeft + x0) + 1, c010 * val.y);
			atomicAdd((tfloat*)(d_volumeweights + (z0 * dimvolume + y1) * dimvolumeft + x0), c010 * weight);

			atomicAdd((tfloat*)(d_volumeft + (z0 * dimvolume + y1) * dimvolumeft + x1), c110 * val.x);
			atomicAdd((tfloat*)(d_volumeft + (z0 * dimvolume + y1) * dimvolumeft + x1) + 1, c110 * val.y);
			atomicAdd((tfloat*)(d_volumeweights + (z0 * dimvolume + y1) * dimvolumeft + x1), c110 * weight);

			atomicAdd((tfloat*)(d_volumeft + (z1 * dimvolume + y0) * dimvolumeft + x0), c001 * val.x);
			atomicAdd((tfloat*)(d_volumeft + (z1 * dimvolume + y0) * dimvolumeft + x0) + 1, c001 * val.y);
			atomicAdd((tfloat*)(d_volumeweights + (z1 * dimvolume + y0) * dimvolumeft + x0), c001 * weight);

			atomicAdd((tfloat*)(d_volumeft + (z1 * dimvolume + y0) * dimvolumeft + x1), c101 * val.x);
			atomicAdd((tfloat*)(d_volumeft + (z1 * dimvolume + y0) * dimvolumeft + x1) + 1, c101 * val.y);
			atomicAdd((tfloat*)(d_volumeweights + (z1 * dimvolume + y0) * dimvolumeft + x1), c101 * weight);

			atomicAdd((tfloat*)(d_volumeft + (z1 * dimvolume + y1) * dimvolumeft + x0), c011 * val.x);
			atomicAdd((tfloat*)(d_volumeft + (z1 * dimvolume + y1) * dimvolumeft + x0) + 1, c011 * val.y);
			atomicAdd((tfloat*)(d_volumeweights + (z1 * dimvolume + y1) * dimvolumeft + x0), c011 * weight);

			atomicAdd((tfloat*)(d_volumeft + (z1 * dimvolume + y1) * dimvolumeft + x1), c111 * val.x);
			atomicAdd((tfloat*)(d_volumeft + (z1 * dimvolume + y1) * dimvolumeft + x1) + 1, c111 * val.y);
			atomicAdd((tfloat*)(d_volumeweights + (z1 * dimvolume + y1) * dimvolumeft + x1), c111 * weight);
		}
	}
}
