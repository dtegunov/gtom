#include "Prerequisites.cuh"
#include "Generics.cuh"
#include "Helper.cuh"


namespace gtom
{
	////////////////////////////
	//CUDA kernel declarations//
	////////////////////////////

	__global__ void Xray2DKernel(tfloat* d_input, tfloat* d_output, int3 dims, tfloat mean, tfloat dev, int region);
	__global__ void Xray3DKernel(tfloat* d_input, tfloat* d_output, int3 dims, tfloat mean, tfloat dev, int region);


	//////////////////////////////////////////////
	//Equivalent of TOM's tom_xraycorrect method//
	//////////////////////////////////////////////

	void d_Xray(tfloat* d_input, tfloat* d_output, int3 dims, tfloat ndev, int region, int batch)
	{
		tfloat* d_scratch;
		if (d_input == d_output)
		{
			cudaMalloc((void**)&d_scratch, Elements(dims) * sizeof(tfloat) * batch);
			cudaMemcpy(d_scratch, d_input, Elements(dims) * sizeof(tfloat) * batch, cudaMemcpyDeviceToDevice);
		}
		else
			d_scratch = d_input;

		imgstats5* d_imagestats;
		cudaMalloc((void**)&d_imagestats, batch * sizeof(imgstats5));
		d_Dev(d_input, d_imagestats, Elements(dims), (char*)NULL, batch);

		imgstats5* h_imagestats = (imgstats5*)MallocFromDeviceArray(d_imagestats, batch * sizeof(imgstats5));

		size_t TpB = min(192, NextMultipleOf(dims.x, 32));
		dim3 grid = dim3((dims.x + TpB - 1) / TpB, dims.y, dims.z);
		for (int b = 0; b < batch; b++)
			if (dims.z <= 1)
			{
				Xray2DKernel << <grid, TpB >> > (d_scratch + Elements(dims) * b,
					d_output + Elements(dims) * b,
					dims,
					h_imagestats[b].mean,
					h_imagestats[b].stddev * ndev,
					region);
			}
			else
			{
				Xray3DKernel << <grid, TpB >> > (d_scratch + Elements(dims) * b,
					d_output + Elements(dims) * b,
					dims,
					h_imagestats[b].mean,
					h_imagestats[b].stddev * ndev,
					region);
			}

		if (d_input == d_output)
			cudaFree(d_scratch);
		cudaFree(d_imagestats);
		free(h_imagestats);
	}


	////////////////
	//CUDA kernels//
	////////////////

	__global__ void Xray2DKernel(tfloat* d_input, tfloat* d_output, int3 dims, tfloat mean, tfloat dev, int region)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= dims.x)
			return;
		int idy = blockIdx.y;

		if (abs(d_input[blockIdx.y * dims.x + idx] - mean) > dev)
		{
			int samples = 0;
			tfloat localmean = (tfloat)0;
			for (int y = max(0, idy - region); y <= min(dims.y - 1, idy + region); y++)
			{
				size_t offsety = y * dims.x;
				for (int x = max(0, idx - region); x <= min(dims.x - 1, idx + region); x++)
				{
					if (x == idx && y == blockIdx.y)
						continue;

					samples++;
					localmean += d_input[offsety + x];
				}
			}

			d_output[blockIdx.y * dims.x + idx] = localmean / (tfloat)samples;
		}
		else
			d_output[blockIdx.y * dims.x + idx] = d_input[blockIdx.y * dims.x + idx];
	}

	__global__ void Xray3DKernel(tfloat* d_input, tfloat* d_output, int3 dims, tfloat mean, tfloat dev, int region)
	{
		uint idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= dims.x)
			return;

		if (abs(d_input[(blockIdx.z * dims.y + blockIdx.y) * dims.x + idx] - mean) > dev)
		{
			uint samples = 0;
			tfloat localmean = (tfloat)0;
			for (int z = max(0, blockIdx.z - region); z <= min(dims.z - 1, blockIdx.z + region); z++)
			{
				for (int y = max(0, blockIdx.y - region); y <= min(dims.y - 1, blockIdx.y + region); y++)
				{
					size_t offsety = (z * dims.y + y) * dims.x;
					for (int x = max(0, idx - region); x <= min(dims.x - 1, idx + region); x++)
					{
						if (x == idx && y == blockIdx.y && z == blockIdx.z)
							continue;

						samples++;
						localmean += d_input[offsety + x];
					}
				}
			}

			d_output[(blockIdx.z * dims.y + blockIdx.y) * dims.x + idx] = localmean / (tfloat)samples;
		}
		else
			d_output[(blockIdx.z * dims.y + blockIdx.y) * dims.x + idx] = d_input[(blockIdx.z * dims.y + blockIdx.y) * dims.x + idx];
	}
}