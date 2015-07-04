#include "Prerequisites.cuh"
#include "Angles.cuh"
#include "Helper.cuh"

namespace gtom
{
	template<uint TpB> __global__ void Project2Dto2DKernel(cudaTex t_volumeRe, cudaTex t_volumeIm, uint dimvolume, tcomplex* d_proj, uint dimproj, uint rmax, uint rmax2);
	template<uint ndims, uint TpB> __global__ void Project3DtoNDKernel(cudaTex t_volumeRe, cudaTex t_volumeIm, uint dimvolume, tcomplex* d_proj, uint dimproj, size_t elementsproj, uint rmax, int rmax2);


	__constant__ float c_matrices[128 * 9];

	void d_rlnProject(cudaTex t_volumeRe, cudaTex t_volumeIm, int3 dimsvolume, tcomplex* d_proj, int3 dimsproj, uint rmax, glm::mat3* d_matrices, uint batch)
	{
		uint ndimsvolume = DimensionCount(dimsvolume);
		uint ndimsproj = DimensionCount(dimsproj);
		if (ndimsvolume < ndimsproj)
			throw;

		rmax = tmin(rmax, dimsproj.x / 2);

		if (ndimsvolume == 3)
		{
			cudaMemcpyToSymbol(c_matrices, d_matrices, batch * sizeof(glm::mat3), 0, cudaMemcpyDeviceToDevice);

			dim3 grid = dim3(1, batch, 1);
			uint elements = ElementsFFT(dimsproj);
			uint TpB = 1 << tmin(7, tmax(7, (uint)(log(elements / 4.0) / log(2.0))));

			if (TpB == 32)
			{
				if (ndimsproj == 2)
					Project3DtoNDKernel<2, 32> <<<grid, 32 >>> (t_volumeRe, t_volumeIm, dimsvolume.x, d_proj, dimsproj.x, elements, rmax, rmax * rmax);
				else if (ndimsproj == 3)
					Project3DtoNDKernel<3, 32> <<<grid, 32 >>> (t_volumeRe, t_volumeIm, dimsvolume.x, d_proj, dimsproj.x, elements, rmax, rmax * rmax);
			}
			else if (TpB == 64)
			{
				if (ndimsproj == 2)
					Project3DtoNDKernel<2, 64> <<<grid, 64 >>> (t_volumeRe, t_volumeIm, dimsvolume.x, d_proj, dimsproj.x, elements, rmax, rmax * rmax);
				else if (ndimsproj == 3)
					Project3DtoNDKernel<3, 64> <<<grid, 64 >>> (t_volumeRe, t_volumeIm, dimsvolume.x, d_proj, dimsproj.x, elements, rmax, rmax * rmax);
			}
			else if (TpB == 128)
			{
				if (ndimsproj == 2)
					Project3DtoNDKernel<2, 128> <<<grid, 128 >>> (t_volumeRe, t_volumeIm, dimsvolume.x, d_proj, dimsproj.x, elements, rmax, rmax * rmax);
				else if (ndimsproj == 3)
					Project3DtoNDKernel<3, 128> <<<grid, 128 >>> (t_volumeRe, t_volumeIm, dimsvolume.x, d_proj, dimsproj.x, elements, rmax, rmax * rmax);
			}
			else if (TpB == 256)
			{
				if (ndimsproj == 2)
					Project3DtoNDKernel<2, 256> <<<grid, 256 >>> (t_volumeRe, t_volumeIm, dimsvolume.x, d_proj, dimsproj.x, elements, rmax, rmax * rmax);
				else if (ndimsproj == 3)
					Project3DtoNDKernel<3, 256> <<<grid, 256 >>> (t_volumeRe, t_volumeIm, dimsvolume.x, d_proj, dimsproj.x, elements, rmax, rmax * rmax);
			}
			else
				throw;
		}
		else
		{			
			cudaMemcpyToSymbol(c_matrices, d_matrices, batch * sizeof(glm::mat3), 0, cudaMemcpyDeviceToDevice);

			dim3 grid = dim3(1, batch, 1);
			uint elements = ElementsFFT(dimsproj);
			uint TpB = 1 << tmin(7, tmax(7, (uint)(log(elements / 4.0) / log(2.0))));
			
			if (TpB == 32)
				Project2Dto2DKernel<32> <<<grid, 32>>> (t_volumeRe, t_volumeIm, dimsvolume.x, d_proj, dimsproj.x, rmax, rmax * rmax);
			else if (TpB == 64)
				Project2Dto2DKernel<64> <<<grid, 64>>> (t_volumeRe, t_volumeIm, dimsvolume.x, d_proj, dimsproj.x, rmax, rmax * rmax);
			else if (TpB == 128)
				Project2Dto2DKernel<128> <<<grid, 128>>> (t_volumeRe, t_volumeIm, dimsvolume.x, d_proj, dimsproj.x, rmax, rmax * rmax);
			else if (TpB == 256)
				Project2Dto2DKernel<256> <<<grid, 256>>> (t_volumeRe, t_volumeIm, dimsvolume.x, d_proj, dimsproj.x, rmax, rmax * rmax);
			else
				throw;
		}
	}

	template<uint TpB> __global__ void __launch_bounds__(TpB) Project2Dto2DKernel(cudaTex t_volumeRe, cudaTex t_volumeIm, uint dimvolume, tcomplex* d_proj, uint dimproj, uint rmax, uint rmax2)
	{
		d_proj += ElementsFFT1(dimproj) * dimproj * blockIdx.y;

		float fx, fy;
		float x0, x1, y0, y1, y, y2, r2;
		bool is_neg_x;
		tcomplex d00, d01, d10, d11, dx0, dx1;

		for (uint id = threadIdx.x; id < ElementsFFT1(dimproj) * dimproj; id += TpB)
		{
			uint idx = id % ElementsFFT1(dimproj);
			uint idy = id / ElementsFFT1(dimproj);

			int x = idx;
			int y = idy <= rmax ? idy : (int)idy - dimproj;
			int r2 = y * y + x * x;
			if (r2 > rmax2)
			{
				d_proj[id] = make_cuComplex(0, 0);
				continue;
			}

			glm::vec2 pos = glm::vec2(x, y);
			pos = glm::vec2(c_matrices[blockIdx.y * 9 + 0] * pos.x + c_matrices[blockIdx.y * 9 + 2] * pos.y,
					c_matrices[blockIdx.y * 9 + 1] * pos.x + c_matrices[blockIdx.y * 9 + 3] * pos.y);

			// Only asymmetric half is stored
			if (pos.x + 1e-5f < 0)
			{
				// Get complex conjugated hermitian symmetry pair
				pos.x = -pos.x;
				pos.y = -pos.y;
				is_neg_x = true;
			}
			else
			{
				is_neg_x = false;
			}

			// Bilinear interpolation (with physical coords)
			// Subtract STARTINGY to accelerate access to data (STARTINGX=0)
			// In that way use DIRECT_A3D_ELEM, rather than A3D_ELEM
			x0 = floor(pos.x);
			fx = pos.x - x0;
			x1 = x0 + 1.5f;
			x0 += 0.5f;

			y0 = floor(pos.y);
			fy = pos.y - y0;
			y0 += dimproj / 2.0f;
			y1 = y0 + 1.5f;
			y0 += 0.5f;

			// Matrix access can be accelerated through pre-calculation of z0*xydim etc.
			d00 = make_cuComplex(tex2D<tfloat>(t_volumeRe, x0, y0), tex2D<tfloat>(t_volumeIm, x0, y0));
			d01 = make_cuComplex(tex2D<tfloat>(t_volumeRe, x1, y0), tex2D<tfloat>(t_volumeIm, x1, y0));
			d10 = make_cuComplex(tex2D<tfloat>(t_volumeRe, x0, y1), tex2D<tfloat>(t_volumeIm, x0, y1));
			d11 = make_cuComplex(tex2D<tfloat>(t_volumeRe, x1, y1), tex2D<tfloat>(t_volumeIm, x1, y1));
			
			// Set the interpolated value in the 2D output array
			dx0 = make_cuComplex(lerp(d00.x, d01.x, fx), lerp(d00.y, d01.y, fx));
			dx1 = make_cuComplex(lerp(d10.x, d11.x, fx), lerp(d10.y, d11.y, fx));

			tcomplex val = make_cuComplex(lerp(dx0.x, dx1.x, fy), lerp(dx0.y, dx1.y, fy));
			if (is_neg_x)
				val.y = -val.y;

			d_proj[id] = val;
		}
	}

	template<uint ndims, uint TpB> __global__ void Project3DtoNDKernel(cudaTex t_volumeRe, cudaTex t_volumeIm, uint dimvolume, tcomplex* d_proj, uint dimproj, size_t elementsproj, uint rmax, int rmax2)
	{
		d_proj += elementsproj * blockIdx.y;

		float x0, x1, y0, y1, z0, z1;
		tcomplex d000, d010, d100, d110, d001, d011, d101, d111, dx00, dx10, dxy0, dx01, dx11, dxy1;

		uint slice = ndims == 3 ? ElementsFFT1(dimproj) * dimproj : 1;
		uint dimft = ElementsFFT1(dimproj);

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
			{
				d_proj[id] = make_cuComplex(0, 0);
				continue;
			}

			glm::vec3 pos = glm::vec3(x, y, z);
			pos = glm::vec3(c_matrices[blockIdx.y * 9 + 0] * pos.x + c_matrices[blockIdx.y * 9 + 3] * pos.y + c_matrices[blockIdx.y * 9 + 6] * pos.z,
					c_matrices[blockIdx.y * 9 + 1] * pos.x + c_matrices[blockIdx.y * 9 + 4] * pos.y + c_matrices[blockIdx.y * 9 + 7] * pos.z,
					c_matrices[blockIdx.y * 9 + 2] * pos.x + c_matrices[blockIdx.y * 9 + 5] * pos.y + c_matrices[blockIdx.y * 9 + 8] * pos.z);

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
			x0 = floor(pos.x + 1e-5f);
			pos.x -= x0;
			x0 += 0.5f;
			x1 = x0 + 1.0f;

			y0 = floor(pos.y);
			pos.y -= y0;
			y0 += (float)(dimvolume / 2) + 0.5f;
			y1 = y0 + 1.0f;

			z0 = floor(pos.z);
			pos.z -= z0;
			z0 += (float)(dimvolume / 2) + 0.5f;
			z1 = z0 + 1.0f;

			// Matrix access can be accelerated through pre-calculation of z0*xydim etc.
			d000 = make_cuComplex(tex3D<tfloat>(t_volumeRe, x0, y0, z0), tex3D<tfloat>(t_volumeIm, x0, y0, z0));
			d001 = make_cuComplex(tex3D<tfloat>(t_volumeRe, x1, y0, z0), tex3D<tfloat>(t_volumeIm, x1, y0, z0));
			d010 = make_cuComplex(tex3D<tfloat>(t_volumeRe, x0, y1, z0), tex3D<tfloat>(t_volumeIm, x0, y1, z0));
			d011 = make_cuComplex(tex3D<tfloat>(t_volumeRe, x1, y1, z0), tex3D<tfloat>(t_volumeIm, x1, y1, z0));
			d100 = make_cuComplex(tex3D<tfloat>(t_volumeRe, x0, y0, z1), tex3D<tfloat>(t_volumeIm, x0, y0, z1));
			d101 = make_cuComplex(tex3D<tfloat>(t_volumeRe, x1, y0, z1), tex3D<tfloat>(t_volumeIm, x1, y0, z1));
			d110 = make_cuComplex(tex3D<tfloat>(t_volumeRe, x0, y1, z1), tex3D<tfloat>(t_volumeIm, x0, y1, z1));
			d111 = make_cuComplex(tex3D<tfloat>(t_volumeRe, x1, y1, z1), tex3D<tfloat>(t_volumeIm, x1, y1, z1));
			
			// Set the interpolated value in the 2D output array
			dx00 = lerp(d000, d001, pos.x);
			dx01 = lerp(d010, d011, pos.x);
			dx10 = lerp(d100, d101, pos.x);
			dx11 = lerp(d110, d111, pos.x);

			dxy0 = lerp(dx00, dx01, pos.y);
			dxy1 = lerp(dx10, dx11, pos.y);
			
			tcomplex val = lerp(dxy0, dxy1, pos.z);

			/*pos.x += 0.5f;
			pos.y += (float)(dimvolume / 2) + 0.5f;
			pos.z += (float)(dimvolume / 2) + 0.5f;
			tcomplex val = make_cuComplex(tex3D<tfloat>(t_volumeRe, pos.x, pos.y, pos.z), tex3D<tfloat>(t_volumeIm, pos.x, pos.y, pos.z));*/

			val.y *= is_neg_x;

			d_proj[id] = val;
		}
	}
}
