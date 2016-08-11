#include "Prerequisites.cuh"
#include "Angles.cuh"
#include "Helper.cuh"
#include "Relion.cuh"
#include "Transformation.cuh"

namespace gtom
{
	template<uint TpB> __global__ void __launch_bounds__(TpB) MagAnisotropyCorrectKernel(cudaTex t_image, uint dimimage, tfloat* d_scaled, uint dimscaled, glm::mat2 transform);
	

	void d_MagAnisotropyCorrect(tfloat* d_image, int2 dimsimage, tfloat* d_scaledimage, int2 dimsscaled, float majorpixel, float minorpixel, float majorangle, uint supersample, uint batch)
	{
		int maxbatch = 32;

		for (int b = 0; b < batch; b += maxbatch)
		{
			int curbatch = tmin(maxbatch, (int)batch - b);

			int2 dimssuper = toInt2(dimsimage.x * supersample + 0, dimsimage.y * supersample + 0);
			tfloat* d_super;
			cudaMalloc((void**)&d_super, Elements2(dimssuper) * curbatch * sizeof(tfloat));

			d_Scale(d_image + Elements2(dimsimage) * b, d_super, toInt3(dimsimage), toInt3(dimssuper), T_INTERP_FOURIER, NULL, NULL, curbatch);

			cudaArray_t a_image;
			cudaTex t_image;

			{
				d_BindTextureTo3DArray(d_super, a_image, t_image, toInt3(dimssuper.x, dimssuper.y, curbatch), cudaFilterModeLinear, false);
			}

			cudaFree(d_super);

			float meanpixel = (majorpixel + minorpixel) * 0.5f;
			majorpixel /= meanpixel;
			minorpixel /= meanpixel;
			glm::mat2 transform = Matrix2Rotation(majorangle) * Matrix2Scale(tfloat2(supersample / majorpixel, supersample / minorpixel)) * Matrix2Rotation(-majorangle);

			uint rmax = dimsscaled.x / 2;

			dim3 grid = dim3((Elements2(dimsscaled) + 127) / 128, curbatch, 1);
			MagAnisotropyCorrectKernel<128> << <grid, 128 >> > (t_image, dimssuper.x, d_scaledimage + Elements2(dimsscaled) * b, dimsscaled.x, transform);

			{
				cudaDestroyTextureObject(t_image);
				cudaFreeArray(a_image);
			}
		}
	}

	template<uint TpB> __global__ void __launch_bounds__(TpB) MagAnisotropyCorrectKernel(cudaTex t_image, uint dimimage, tfloat* d_scaled, uint dimscaled, glm::mat2 transform)
	{
		float fx, fy;
		float x0, x1, y0, y1;
		tfloat d00, d01, d10, d11, dx0, dx1;

		d_scaled += dimscaled * dimscaled * blockIdx.y;
		float zcoord = blockIdx.y + 0.5f;
		int imagecenter = dimimage / 2;

		for (uint id = blockIdx.x * blockDim.x + threadIdx.x; id < dimscaled * dimscaled; id += blockDim.x * TpB)
		{
			uint idx = id % dimscaled;
			uint idy = id / dimscaled;

			int x = (int)idx - dimscaled / 2;
			int y = (int)idy - dimscaled / 2;

			glm::vec2 pos = transform * glm::vec2(x, y);

			pos.x += imagecenter;
			pos.y += imagecenter;

			// Bilinear interpolation (with physical coords)
			x0 = floor(pos.x);
			fx = pos.x - x0;
			x1 = x0 + 1.5f;
			x0 += 0.5f;

			y0 = floor(pos.y);
			fy = pos.y - y0;
			y1 = y0 + 1.5f;
			y1 += 0.5f;

			d00 = tex3D<tfloat>(t_image, x0, y0, zcoord);
			d01 = tex3D<tfloat>(t_image, x1, y0, zcoord);
			d10 = tex3D<tfloat>(t_image, x0, y1, zcoord);
			d11 = tex3D<tfloat>(t_image, x1, y1, zcoord);

			dx0 = lerp(d00, d01, fx);
			dx1 = lerp(d10, d11, fx);

			tfloat val = lerp(dx0, dx1, fy);

			d_scaled[id] = val;
		}
	}
}
