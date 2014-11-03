#include "Prerequisites.cuh"
#include "FFT.cuh"

////////////////////////////
//CUDA kernel declarations//
////////////////////////////

__global__ void FFTCropEvenKernel(tcomplex* d_input, tcomplex* d_output, int3 olddims, int3 newdims);
__global__ void FFTCropOddKernel(tcomplex* d_input, tcomplex* d_output, int3 olddims, int3 newdims);
template <class T> __global__ void FFTFullCropKernel(T* d_input, T* d_output, int3 olddims, int3 newdims);
__global__ void FFTPadEvenKernel(tcomplex* d_input, tcomplex* d_output, int3 olddims, int3 newdims);
__global__ void FFTFullPadEvenKernel(tcomplex* d_input, tcomplex* d_output, int3 olddims, int3 newdims);

__host__ __device__ tcomplex toTcomplex(tfloat r, tfloat i)
{
	tcomplex value = {r, i};
	return value;
}


////////////////
//Host methods//
////////////////

void d_FFTCrop(tcomplex* d_input, tcomplex* d_output, int3 olddims, int3 newdims, int batch)
{
	size_t elementsnew = ElementsFFT(newdims);
	size_t elementsold = ElementsFFT(olddims);

	int TpB = min(256, NextMultipleOf(newdims.x / 2 + 1, 32));
	dim3 grid = dim3(newdims.y, newdims.z, batch);
	if(newdims.x % 2 == 0)
		FFTCropEvenKernel <<<grid, TpB>>> (d_input, d_output, olddims, newdims);
	else
		FFTCropOddKernel <<<grid, TpB>>> (d_input, d_output, olddims, newdims);
}

template <class T> void d_FFTFullCrop(T* d_input, T* d_output, int3 olddims, int3 newdims, int batch)
{
	size_t elementsnew = Elements(newdims);
	size_t elementsold = Elements(olddims);

	int TpB = min(256, NextMultipleOf(newdims.x, 32));
	dim3 grid = dim3(newdims.y, newdims.z, batch);
	FFTFullCropKernel <<<grid, TpB>>> (d_input, d_output, olddims, newdims);
	cudaStreamQuery(0);
}
template void d_FFTFullCrop<tcomplex>(tcomplex* d_input, tcomplex* d_output, int3 olddims, int3 newdims, int batch);
template void d_FFTFullCrop<tfloat>(tfloat* d_input, tfloat* d_output, int3 olddims, int3 newdims, int batch);

void d_FFTPad(tcomplex* d_input, tcomplex* d_output, int3 olddims, int3 newdims, int batch)
{
	size_t elementsnew = ElementsFFT(newdims);
	size_t elementsold = ElementsFFT(olddims);

	int TpB = min(256, NextMultipleOf(newdims.x / 2 + 1, 32));
	dim3 grid = dim3(newdims.y, newdims.z, batch);
	FFTPadEvenKernel <<<grid, TpB>>> (d_input, d_output, olddims, newdims);
}

void d_FFTFullPad(tcomplex* d_input, tcomplex* d_output, int3 olddims, int3 newdims, int batch)
{
	size_t elementsnew = Elements(newdims);
	size_t elementsold = Elements(olddims);

	int TpB = min(256, NextMultipleOf(newdims.x, 32));
	dim3 grid = dim3(newdims.y, newdims.z, batch);
	FFTFullPadEvenKernel <<<grid, TpB>>> (d_input, d_output, olddims, newdims);
}


////////////////
//CUDA kernels//
////////////////

__global__ void FFTCropEvenKernel(tcomplex* d_input, tcomplex* d_output, int3 olddims, int3 newdims)
{
	d_input += ElementsFFT(olddims) * blockIdx.z;
	d_output += ElementsFFT(newdims) * blockIdx.z;

	for(int x = threadIdx.x; x < newdims.x / 2 + 1; x += blockDim.x)
	{
		int newry = ((blockIdx.x + (newdims.y + 1) / 2) % newdims.y);
		int newrz = ((blockIdx.y + (newdims.z + 1) / 2) % newdims.z);

		int oldry = (olddims.y - newdims.y + ((olddims.y & 1 - (newdims.y & 1)) % 2)) / 2 + newry;
		int oldrz = (olddims.z - newdims.z + ((olddims.z & 1 - (newdims.z & 1)) % 2)) / 2 + newrz;

		int oldy = ((oldry + (olddims.y) / 2) % olddims.y);
		int oldz = ((oldrz + (olddims.z) / 2) % olddims.z);

		if(x == newdims.x / 2)
		{
			if(oldy != 0)
				oldy = olddims.y - oldy;
			if(oldz != 0)
				oldz = olddims.z - oldz;
		}
		if(blockIdx.x == newdims.y / 2)
			oldy = newdims.y / 2;
		if(blockIdx.y == newdims.z / 2)
			oldz = newdims.z / 2;

		tcomplex val = d_input[(oldz * olddims.y + oldy) * (olddims.x / 2 + 1) + x];
		if(x == newdims.x / 2)
			d_output[(blockIdx.y * newdims.y + blockIdx.x) * (newdims.x / 2 + 1) + x] = cconj(val);
		else
			d_output[(blockIdx.y * newdims.y + blockIdx.x) * (newdims.x / 2 + 1) + x] = val;
	}
}

__global__ void FFTCropOddKernel(tcomplex* d_input, tcomplex* d_output, int3 olddims, int3 newdims)
{
	d_input += ElementsFFT(olddims) * blockIdx.z;
	d_output += ElementsFFT(newdims) * blockIdx.z;

	for(int x = threadIdx.x; x < newdims.x / 2 + 1; x += blockDim.x)
	{
		int newry = ((blockIdx.x + (newdims.y + 1) / 2) % newdims.y);
		int newrz = ((blockIdx.y + (newdims.z + 1) / 2) % newdims.z);

		int oldry = (olddims.y - newdims.y + ((olddims.y & 1 - (newdims.y & 1)) % 2)) / 2 + newry;
		int oldrz = (olddims.z - newdims.z + ((olddims.z & 1 - (newdims.z & 1)) % 2)) / 2 + newrz;

		int oldy = ((oldry + (olddims.y) / 2) % olddims.y);
		int oldz = ((oldrz + (olddims.z) / 2) % olddims.z);
		if(blockIdx.x == newdims.y / 2)
			oldy = newdims.y / 2;
		if(blockIdx.y == newdims.z / 2)
			oldz = newdims.z / 2;

		d_output[(blockIdx.y * newdims.y + blockIdx.x) * (newdims.x / 2 + 1) + x] = d_input[(oldz * olddims.y + oldy) * (olddims.x / 2 + 1) + x];
	}
}

template <class T> __global__ void FFTFullCropKernel(T* d_input, T* d_output, int3 olddims, int3 newdims)
{
	int oldy = blockIdx.x;
	if (oldy >= newdims.y / 2)
		oldy += olddims.y - newdims.y;
	int oldz = blockIdx.y;
	if (oldz >= newdims.z / 2)
		oldz += olddims.z - newdims.z;

	d_input += Elements(olddims) * blockIdx.z + (oldz * olddims.y + oldy) * olddims.x;
	d_output += Elements(newdims) * blockIdx.z + (blockIdx.y * newdims.y + blockIdx.x) * newdims.x;

	for(int x = threadIdx.x; x < newdims.x; x += blockDim.x)
	{
		int oldx = x;
		if (oldx >= newdims.x / 2)
			oldx += olddims.x - newdims.x;

		d_output[x] = d_input[oldx];
	}
}

__global__ void FFTPadEvenKernel(tcomplex* d_input, tcomplex* d_output, int3 olddims, int3 newdims)
{
	d_input += ElementsFFT(olddims) * blockIdx.z;
	d_output += ElementsFFT(newdims) * blockIdx.z;

	for(int x = threadIdx.x; x < newdims.x / 2 + 1; x += blockDim.x)
	{
		int newry = ((blockIdx.x + newdims.y / 2) % newdims.y);
		int newrz = ((blockIdx.y + newdims.z / 2) % newdims.z);

		int oldry =  newry + (olddims.y - newdims.y) / 2;
		int oldrz =  newrz + (olddims.z - newdims.z) / 2;
	
		if(x < (olddims.x + 1) / 2 && oldry >= 0 && oldry < olddims.y && oldrz >= 0 && oldrz < olddims.z)
		{
			int oldy = ((oldry + (olddims.y + 1) / 2) % olddims.y);
			int oldz = ((oldrz + (olddims.z + 1) / 2) % olddims.z);

			d_output[(blockIdx.y * newdims.y + blockIdx.x) * (newdims.x / 2 + 1) + x] = d_input[(oldz * olddims.y + oldy) * (olddims.x / 2 + 1) + x];
		}
		else
			d_output[(blockIdx.y * newdims.y + blockIdx.x) * (newdims.x / 2 + 1) + x] = toTcomplex(0.0f, 0.0f);
	}
}

__global__ void FFTFullPadEvenKernel(tcomplex* d_input, tcomplex* d_output, int3 olddims, int3 newdims)
{
	d_input += Elements(olddims) * blockIdx.z;
	d_output += Elements(newdims) * blockIdx.z;

	for(int x = threadIdx.x; x < newdims.x; x += blockDim.x)
	{
		int newrx = ((x + (newdims.x) / 2) % newdims.x);
		int newry = ((blockIdx.x + (newdims.y) / 2) % newdims.y);
		int newrz = ((blockIdx.y + (newdims.z) / 2) % newdims.z);
	
		int oldrx =  newrx + (olddims.x - newdims.x - ((olddims.x & 1 - (newdims.x & 1)) % 2)) / 2;
		int oldry =  newry + (olddims.y - newdims.y - ((olddims.y & 1 - (newdims.y & 1)) % 2)) / 2;
		int oldrz =  newrz + (olddims.z - newdims.z - ((olddims.z & 1 - (newdims.z & 1)) % 2)) / 2;
	
		if(oldrx >= 0 && oldrx < olddims.x && oldry >= 0 && oldry < olddims.y && oldrz >= 0 && oldrz < olddims.z)
		{
			int oldx = ((oldrx + (olddims.x + 1) / 2) % olddims.x);
			int oldy = ((oldry + (olddims.y + 1) / 2) % olddims.y);
			int oldz = ((oldrz + (olddims.z + 1) / 2) % olddims.z);

			d_output[(blockIdx.y * newdims.y + blockIdx.x) * newdims.x + x] = d_input[(oldz * olddims.y + oldy) * olddims.x + oldx];
		}
		else
			d_output[(blockIdx.y * newdims.y + blockIdx.x) * newdims.x + x] = toTcomplex(0.0f, 0.0f);
	}
}