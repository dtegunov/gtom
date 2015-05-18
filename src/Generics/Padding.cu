#include "Prerequisites.cuh"
#include "Generics.cuh"


namespace gtom
{
	////////////////////////////
	//CUDA kernel declarations//
	////////////////////////////

	template <class T> __global__ void PadValueKernel(T* d_input, T* d_output, int3 inputdims, int3 outputdims, int3 offset, T value);
	template <class T> __global__ void PadMirrorKernel(T* d_input, T* d_output, int3 inputdims, int3 outputdims, int3 offset);
	template <class T> __global__ void PadTileKernel(T* d_input, T* d_output, int3 inputdims, int3 outputdims, int3 offset);


	/////////////////////////////////////////////////////////////////////
	//Extract a portion of 1/2/3-dimensional data with cyclic boudaries//
	/////////////////////////////////////////////////////////////////////

	template <class T> void d_Pad(T* d_input, T* d_output, int3 inputdims, int3 outputdims, T_PAD_MODE mode, T value, int batch)
	{
		int3 inputcenter = toInt3(inputdims.x / 2, inputdims.y / 2, inputdims.z / 2);
		int3 outputcenter = toInt3(outputdims.x / 2, outputdims.y / 2, outputdims.z / 2);

		size_t TpB = min(256, NextMultipleOf(outputdims.x, 32));
		dim3 grid = dim3((outputdims.x + TpB - 1) / TpB, outputdims.y, outputdims.z);

		int3 offset = toInt3(inputcenter.x - outputcenter.x, inputcenter.y - outputcenter.y, inputcenter.z - outputcenter.z);

		for (int b = 0; b < batch; b++)
			if (mode == T_PAD_VALUE)
				PadValueKernel << <grid, (int)TpB >> > (d_input + Elements(inputdims) * b, d_output + Elements(outputdims) * b, inputdims, outputdims, offset, value);
			else if (mode == T_PAD_MIRROR)
				PadMirrorKernel << <grid, (int)TpB >> > (d_input + Elements(inputdims) * b, d_output + Elements(outputdims) * b, inputdims, outputdims, offset);
			else if (mode == T_PAD_TILE)
				PadTileKernel << <grid, (int)TpB >> > (d_input + Elements(inputdims) * b, d_output + Elements(outputdims) * b, inputdims, outputdims, offset);
			cudaStreamQuery(0);
	}
	template void d_Pad<int>(int* d_input, int* d_output, int3 inputdims, int3 outputdims, T_PAD_MODE mode, int value, int batch);
	template void d_Pad<float>(float* d_input, float* d_output, int3 inputdims, int3 outputdims, T_PAD_MODE mode, float value, int batch);
	template void d_Pad<double>(double* d_input, double* d_output, int3 inputdims, int3 outputdims, T_PAD_MODE mode, double value, int batch);


	////////////////
	//CUDA kernels//
	////////////////

	template <class T> __global__ void PadValueKernel(T* d_input, T* d_output, int3 inputdims, int3 outputdims, int3 offset, T value)
	{
		int idy = blockIdx.y;
		int idz = blockIdx.z;

		bool outofbounds = false;

		int oy, oz, ox;

		oy = offset.y + idy;
		if (oy < 0 || oy >= inputdims.y)
		{
			outofbounds = true;
		}
		else
		{
			oz = offset.z + idz;
			if (oz < 0 || oz >= inputdims.z)
				outofbounds = true;
		}

		T* offsetoutput = d_output + (idz * outputdims.y + idy) * outputdims.x;

		for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < outputdims.x; idx += blockDim.x * gridDim.x)
		{
			if (outofbounds)
				offsetoutput[idx] = value;
			else
			{
				ox = offset.x + idx;
				if (ox < 0 || ox >= inputdims.x)
					offsetoutput[idx] = value;
				else
					offsetoutput[idx] = d_input[(oz * inputdims.y + oy) * inputdims.x + ox];
			}
		}
	}

	template <class T> __global__ void PadMirrorKernel(T* d_input, T* d_output, int3 inputdims, int3 outputdims, int3 offset)
	{
		int idy = blockIdx.y;
		int idz = blockIdx.z;

		uint ox = 0, oy = 0, oz = 0;
		if (inputdims.y > 1)
		{
			oy = (uint)(offset.y + idy + inputdims.y * 99998) % (uint)(inputdims.y * 2);
			if (oy >= inputdims.y)
				oy = inputdims.y * 2 - 1 - oy;
		}
		if (inputdims.z > 1)
		{
			oz = (uint)(offset.z + idz + inputdims.z * 99998) % (uint)(inputdims.z * 2);
			if (oz >= inputdims.z)
				oz = inputdims.z * 2 - 1 - oz;
		}

		T* offsetoutput = d_output + (idz * outputdims.y + idy) * outputdims.x;
		T* offsetinput = d_input + (oz * inputdims.y + oy) * inputdims.x;

		for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < outputdims.x; idx += blockDim.x * gridDim.x)
		{
			ox = (uint)(offset.x + idx + inputdims.x * 99998) % (uint)(inputdims.x * 2);
			if (ox >= inputdims.x)
				ox = inputdims.x * 2 - 1 - ox;
			offsetoutput[idx] = offsetinput[ox];
		}
	}

	template <class T> __global__ void PadTileKernel(T* d_input, T* d_output, int3 inputdims, int3 outputdims, int3 offset)
	{
		int idy = blockIdx.y;
		int idz = blockIdx.z;

		int oy = (offset.y + idy + inputdims.y * 99999) % inputdims.y;
		int oz = (offset.z + idz + inputdims.z * 99999) % inputdims.z;

		T* offsetoutput = d_output + (idz * outputdims.y + idy) * outputdims.x;
		T* offsetinput = d_input + (oz * inputdims.y + oy) * inputdims.x;

		for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < outputdims.x; idx += blockDim.x * gridDim.x)
			offsetoutput[idx] = offsetinput[(offset.x + idx + inputdims.x * 99999) % inputdims.x];
	}
}