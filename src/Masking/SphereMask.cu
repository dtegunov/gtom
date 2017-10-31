#include "Prerequisites.cuh"


namespace gtom
{
	////////////////////////////
	//CUDA kernel declarations//
	////////////////////////////

	template <class T> __global__ void SphereMaskKernel(T* d_input, T* d_output, int3 size, tfloat radius, tfloat sigma, tfloat3 center);
	__global__ void SphereMaskFTKernel(tfloat* d_input, tfloat* d_output, int3 dims, int radius2);


	////////////////
	//Host methods//
	////////////////

	template <class T> void d_SphereMask(T* d_input,
										T* d_output,
										int3 size,
										tfloat* radius,
										tfloat sigma,
										tfloat3* center,
										int batch)
	{
		tfloat _radius = radius != NULL ? *radius : min(min(size.x, size.y), size.z > 1 ? size.z : size.x) / 2;
		tfloat3 _center = center != NULL ? *center : tfloat3(size.x / 2, size.y / 2, size.z / 2);

		int TpB = 256;
		dim3 grid = dim3(size.y, size.z, batch);
		SphereMaskKernel<T> << <grid, TpB >> > (d_input, d_output, size, _radius, sigma, _center);
	}
	template void d_SphereMask<tfloat>(tfloat* d_input, tfloat* d_output, int3 size, tfloat* radius, tfloat sigma, tfloat3* center, int batch);
	template void d_SphereMask<tcomplex>(tcomplex* d_input, tcomplex* d_output, int3 size, tfloat* radius, tfloat sigma, tfloat3* center, int batch);

	void d_SphereMaskFT(tfloat* d_input, tfloat* d_output, int3 dims, int radius, uint batch)
	{
		int TpB = tmin(128, NextMultipleOf(dims.x, 32));
		dim3 grid = dim3(dims.y, dims.z, batch);
		SphereMaskFTKernel <<<grid, TpB>>> (d_input, d_output, dims, radius * radius);
	}


	////////////////
	//CUDA kernels//
	////////////////

	template <class T> __global__ void SphereMaskKernel(T* d_input, T* d_output, int3 size, tfloat radius, tfloat sigma, tfloat3 center)
	{
		if (threadIdx.x >= size.x)
			return;

		//For batch mode
		int offset = blockIdx.z * size.x * size.y * size.z + blockIdx.y * size.x * size.y + blockIdx.x * size.x;

		tfloat xsq, ysq, zsq, length;
		T maskvalue;

		//Squared y and z distance from center
		ysq = (tfloat)blockIdx.x - center.y;
		ysq *= ysq;
		if (size.z > 1)
		{
			zsq = (tfloat)blockIdx.y - center.z;
			zsq *= zsq;
		}
		else
			zsq = 0;

		for (int x = threadIdx.x; x < size.x; x += blockDim.x)
		{
			xsq = (tfloat)x - center.x;
			xsq *= xsq;
			//Distance from center
			length = sqrt(xsq + ysq + zsq);

			if (length < radius)
				maskvalue = 1;
			else
			{
				//Smooth border
				if (sigma > (tfloat)0)
				{
					maskvalue = tmax(0, (cos(tmin(1.0f, (length - radius) / sigma) * PI) + 1.0f) * 0.5f);
				}
				//Hard border
				else
					maskvalue = 0;
			}

			//Write masked input to output
			d_output[offset + x] = maskvalue * d_input[offset + x];
		}
	}

	template<> __global__ void SphereMaskKernel<tcomplex>(tcomplex* d_input, tcomplex* d_output, int3 size, tfloat radius, tfloat sigma, tfloat3 center)
	{
		if (threadIdx.x >= size.x)
			return;

		//For batch mode
		int offset = blockIdx.z * size.x * size.y * size.z + blockIdx.y * size.x * size.y + blockIdx.x * size.x;

		tfloat xsq, ysq, zsq, length;
		tfloat maskvalue;

		//Squared y and z distance from center
		ysq = (tfloat)(blockIdx.x + 1) - center.y;
		ysq *= ysq;
		if (size.z > 1)
		{
			zsq = (tfloat)(blockIdx.y + 1) - center.z;
			zsq *= zsq;
		}
		else
			zsq = 0;

		for (int x = threadIdx.x; x < size.x; x += blockDim.x)
		{
			xsq = (tfloat)(x + 1) - center.x;
			xsq *= xsq;
			//Distance from center
			length = sqrt(xsq + ysq + zsq);

			if (length < radius)
				maskvalue = 1;
			else
			{
				//Smooth border
				if (sigma > (tfloat)0)
				{
					maskvalue = (cos(tmin(1.0f, (length - radius) / sigma) * PI) + 1.0f) * 0.5f;
					if (maskvalue < (tfloat)0.1353)
						maskvalue = 0;
				}
				//Hard border
				else
					maskvalue = 0;
			}

			//Write masked input to output
			d_output[offset + x].x = maskvalue * d_input[offset + x].x;
			d_output[offset + x].y = maskvalue * d_input[offset + x].y;
		}
	}

	__global__ void SphereMaskFTKernel(tfloat* d_input, tfloat* d_output, int3 dims, int radius2)
	{
		int z = blockIdx.y;
		int y = blockIdx.x;

		d_input +=  blockIdx.z * ElementsFFT(dims) + (z * dims.y + y) * (dims.x / 2 + 1);
		d_output += blockIdx.z * ElementsFFT(dims) + (z * dims.y + y) * (dims.x / 2 + 1);

		int zp = z < dims.z / 2 + 1 ? z : z - dims.x;
		zp *= zp;
		int yp = y < dims.y / 2 + 1 ? y : y - dims.x;
		yp *= yp;

		for (int x = threadIdx.x; x < dims.x / 2 + 1; x += blockDim.x)
		{
			int r = x * x + yp + zp;

			if (r < radius2)
				d_output[x] = d_input[x];
			else
				d_output[x] = 0;
		}
	}
}