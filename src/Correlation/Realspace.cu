#include "Prerequisites.cuh"

namespace gtom
{
	//////////////////////////////
	// CUDA kernel declarations //
	//////////////////////////////

	__global__ void CorrelateRealspaceKernel(tfloat* d_image, tfloat* d_sum1, tfloat* d_sum2, uint2 dimsimage, tfloat* d_template, tfloat* d_mask, uint2 dimstemplate, tfloat* d_samples, tfloat* d_corr);


	////////////////////////////////////////////////////////////////
	// Performs masked, locally normalized real-space correlation //
	////////////////////////////////////////////////////////////////

	void d_CorrelateRealspace(tfloat* d_image, tfloat* d_sum1, tfloat* d_sum2, int2 dimsimage, tfloat* d_template, tfloat* d_mask, int2 dimstemplate, tfloat* d_samples, tfloat* d_corr, uint nimages)
	{
		dim3 TpB = dim3(32, 4);
		dim3 grid = dim3(dimsimage.x, dimsimage.y, nimages);
		CorrelateRealspaceKernel <<<grid, TpB>>> (d_image, d_sum1, d_sum2, make_uint2(dimsimage.x, dimsimage.y), d_template, d_mask, make_uint2(dimstemplate.x, dimstemplate.y), d_samples, d_corr);
	}


	//////////////////
	// CUDA kernels //
	//////////////////

	__global__ void CorrelateRealspaceKernel(tfloat* d_image, tfloat* d_sum1, tfloat* d_sum2, uint2 dimsimage, tfloat* d_template, tfloat* d_mask, uint2 dimstemplate, tfloat* d_samples, tfloat* d_corr)
	{
		__shared__ tfloat s_corr[128];

		d_image += Elements2(dimsimage) * blockIdx.z;
		d_sum1 += Elements2(dimsimage) * blockIdx.z;
		d_sum2 += Elements2(dimsimage) * blockIdx.z;
		d_corr += Elements2(dimsimage) * blockIdx.z;
		d_template += Elements2(dimstemplate) * blockIdx.z;
		d_mask += Elements2(dimstemplate) * blockIdx.z;

		uint idx = blockIdx.x;
		uint idy = blockIdx.y;

		tfloat sum1 = d_sum1[idy * dimsimage.x + idx];
		tfloat sum2 = d_sum2[idy * dimsimage.x + idx];
		tfloat samples = d_samples[blockIdx.z];
		tfloat mean = sum1 / samples;
		tfloat std = sqrt((samples * sum2 - (sum1 * sum1))) / samples;
		if (abs(std) > (tfloat)1e-5)
			std = (tfloat)1 / std;
		else
			std = 0;
		tfloat corr = 0;

		uint2 halftemplate = make_uint2(dimstemplate.x / 2, dimstemplate.y / 2);

		for (uint y = threadIdx.y; y < dimstemplate.y; y += 4)
		{
			uint yy = (idy + y + dimsimage.y - halftemplate.y) % dimsimage.y;

			for (uint x = threadIdx.x; x < dimstemplate.x; x += 32)
			{
				uint xx = (idx + x + dimsimage.x - halftemplate.x) % dimsimage.x;

				tfloat val = (d_image[yy * dimsimage.x + xx] - mean) * std;
				tfloat mask = d_mask[y * dimstemplate.x + x];
				val *= mask;

				tfloat templateval = d_template[y * dimstemplate.x + x];
				corr += val * templateval;
			}
		}
		s_corr[threadIdx.y * 32 + threadIdx.x] = corr;
		__syncthreads();

		/*if (threadIdx.y == 0)
		{
			for (int i = threadIdx.x + 32; i < 128; i += 32)
				corr += s_corr[i];
			s_corr[threadIdx.x] = corr;
		}
		__syncthreads();*/

		if (threadIdx.x == 0 && threadIdx.y == 0)
		{
			for (int i = 1; i < 128; i++)
				corr += s_corr[i];

			d_corr[idy * dimsimage.x + idx] = corr / samples;
		}
	}
}