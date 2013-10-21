#include "../Prerequisites.cuh"
#include "../Functions.cuh"

////////////////////////////
//CUDA kernel declarations//
////////////////////////////

__global__ void FFTCropEvenKernel(tcomplex* d_input, tcomplex* d_output, int3 olddims, int3 newdims);
__global__ void FFTCropOddKernel(tcomplex* d_input, tcomplex* d_output, int3 olddims, int3 newdims);
__global__ void FFTFullCropKernel(tcomplex* d_input, tcomplex* d_output, int3 olddims, int3 newdims);
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
	size_t elementsnew = (newdims.x / 2 + 1) * newdims.y * newdims.z;
	size_t elementsold = (olddims.x / 2 + 1) * olddims.y * olddims.z;

	int TpB = min(256, NextMultipleOf(newdims.x / 2 + 1, 32));
	dim3 grid = dim3((newdims.x / 2 + 1 + TpB - 1) / TpB, newdims.y, newdims.z);
	if(newdims.x % 2 == 0)
		for(int b = 0; b < batch; b++)
			FFTCropEvenKernel <<<grid, TpB>>> (d_input + elementsold * b, d_output + elementsnew * b, olddims, newdims);
	else
		for(int b = 0; b < batch; b++)
			FFTCropOddKernel <<<grid, TpB>>> (d_input + elementsold * b, d_output + elementsnew * b, olddims, newdims);

	cudaDeviceSynchronize();
}

void d_FFTFullCrop(tcomplex* d_input, tcomplex* d_output, int3 olddims, int3 newdims, int batch)
{
	size_t elementsnew = newdims.x * newdims.y * newdims.z;
	size_t elementsold = olddims.x * olddims.y * olddims.z;

	int TpB = min(256, NextMultipleOf(newdims.x, 32));
	dim3 grid = dim3((newdims.x + TpB - 1) / TpB, newdims.y, newdims.z);
	for(int b = 0; b < batch; b++)
		FFTFullCropKernel <<<grid, TpB>>> (d_input + elementsold * b, d_output + elementsnew * b, olddims, newdims);

	cudaDeviceSynchronize();
}

void d_FFTPad(tcomplex* d_input, tcomplex* d_output, int3 olddims, int3 newdims, int batch)
{
	size_t elementsnew = (newdims.x / 2 + 1) * newdims.y * newdims.z;
	size_t elementsold = (olddims.x / 2 + 1) * olddims.y * olddims.z;

	int TpB = min(256, NextMultipleOf(newdims.x / 2 + 1, 32));
	dim3 grid = dim3((newdims.x / 2 + 1 + TpB - 1) / TpB, newdims.y, newdims.z);
	for(int b = 0; b < batch; b++)
		FFTPadEvenKernel <<<grid, TpB>>> (d_input + elementsold * b, d_output + elementsnew * b, olddims, newdims);

	cudaDeviceSynchronize();
}

void d_FFTFullPad(tcomplex* d_input, tcomplex* d_output, int3 olddims, int3 newdims, int batch)
{
	size_t elementsnew = newdims.x * newdims.y * newdims.z;
	size_t elementsold = olddims.x * olddims.y * olddims.z;

	int TpB = min(256, NextMultipleOf(newdims.x, 32));
	dim3 grid = dim3((newdims.x + TpB - 1) / TpB, newdims.y, newdims.z);
	for(int b = 0; b < batch; b++)
		FFTFullPadEvenKernel <<<grid, TpB>>> (d_input + elementsold * b, d_output + elementsnew * b, olddims, newdims);

	cudaDeviceSynchronize();
}


////////////////
//CUDA kernels//
////////////////

__global__ void FFTCropEvenKernel(tcomplex* d_input, tcomplex* d_output, int3 olddims, int3 newdims)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if(x >= newdims.x / 2 + 1)
		return;

	int newry = ((blockIdx.y + (newdims.y + 1) / 2) % newdims.y);
	int newrz = ((blockIdx.z + (newdims.z + 1) / 2) % newdims.z);

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
	if(blockIdx.y == newdims.y / 2)
		oldy = newdims.y / 2;
	if(blockIdx.z == newdims.z / 2)
		oldz = newdims.z / 2;

	tcomplex val = d_input[(oldz * olddims.y + oldy) * (olddims.x / 2 + 1) + x];
	if(x == newdims.x / 2)
		d_output[(blockIdx.z * newdims.y + blockIdx.y) * (newdims.x / 2 + 1) + x] = cconj(val);
	else
		d_output[(blockIdx.z * newdims.y + blockIdx.y) * (newdims.x / 2 + 1) + x] = val;
}

__global__ void FFTCropOddKernel(tcomplex* d_input, tcomplex* d_output, int3 olddims, int3 newdims)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if(x >= newdims.x / 2 + 1)
		return;

	int newry = ((blockIdx.y + (newdims.y + 1) / 2) % newdims.y);
	int newrz = ((blockIdx.z + (newdims.z + 1) / 2) % newdims.z);

	int oldry = (olddims.y - newdims.y + ((olddims.y & 1 - (newdims.y & 1)) % 2)) / 2 + newry;
	int oldrz = (olddims.z - newdims.z + ((olddims.z & 1 - (newdims.z & 1)) % 2)) / 2 + newrz;

	int oldy = ((oldry + (olddims.y) / 2) % olddims.y);
	int oldz = ((oldrz + (olddims.z) / 2) % olddims.z);
	if(blockIdx.y == newdims.y / 2)
		oldy = newdims.y / 2;
	if(blockIdx.z == newdims.z / 2)
		oldz = newdims.z / 2;

	d_output[(blockIdx.z * newdims.y + blockIdx.y) * (newdims.x / 2 + 1) + x] = d_input[(oldz * olddims.y + oldy) * (olddims.x / 2 + 1) + x];
}

__global__ void FFTFullCropKernel(tcomplex* d_input, tcomplex* d_output, int3 olddims, int3 newdims)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if(x >= newdims.x)
		return;
	
	int newrx = ((x + (newdims.x + 1) / 2) % newdims.x);
	int newry = ((blockIdx.y + (newdims.y + 1) / 2) % newdims.y);
	int newrz = ((blockIdx.z + (newdims.z + 1) / 2) % newdims.z);
	
	int oldrx = (olddims.x - newdims.x + ((olddims.x & 1 - (newdims.x & 1)) % 2)) / 2 + newrx;
	int oldry = (olddims.y - newdims.y + ((olddims.y & 1 - (newdims.y & 1)) % 2)) / 2 + newry;
	int oldrz = (olddims.z - newdims.z + ((olddims.z & 1 - (newdims.z & 1)) % 2)) / 2 + newrz;
	
	int oldx = ((oldrx + (olddims.x) / 2) % olddims.x);
	int oldy = ((oldry + (olddims.y) / 2) % olddims.y);
	int oldz = ((oldrz + (olddims.z) / 2) % olddims.z);
	if(x == newdims.x / 2)
		oldx = newdims.x / 2;
	if(blockIdx.y == newdims.y / 2)
		oldy = newdims.y / 2;
	if(blockIdx.z == newdims.z / 2)
		oldz = newdims.z / 2;

	d_output[(blockIdx.z * newdims.y + blockIdx.y) * newdims.x + x] = d_input[(oldz * olddims.y + oldy) * olddims.x + oldx];
}

__global__ void FFTPadEvenKernel(tcomplex* d_input, tcomplex* d_output, int3 olddims, int3 newdims)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if(x >= newdims.x / 2 + 1)
		return;

	int newry = ((blockIdx.y + newdims.y / 2) % newdims.y);
	int newrz = ((blockIdx.z + newdims.z / 2) % newdims.z);

	int oldry =  newry + (olddims.y - newdims.y) / 2;
	int oldrz =  newrz + (olddims.z - newdims.z) / 2;
	
	if(x < (olddims.x + 1) / 2 && oldry >= 0 && oldry < olddims.y && oldrz >= 0 && oldrz < olddims.z)
	{
		int oldy = ((oldry + (olddims.y + 1) / 2) % olddims.y);
		int oldz = ((oldrz + (olddims.z + 1) / 2) % olddims.z);

		d_output[(blockIdx.z * newdims.y + blockIdx.y) * (newdims.x / 2 + 1) + x] = d_input[(oldz * olddims.y + oldy) * (olddims.x / 2 + 1) + x];
	}
	else
		d_output[(blockIdx.z * newdims.y + blockIdx.y) * (newdims.x / 2 + 1) + x] = toTcomplex((tfloat)0, (tfloat)0);
}

__global__ void FFTFullPadEvenKernel(tcomplex* d_input, tcomplex* d_output, int3 olddims, int3 newdims)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if(x >= newdims.x)
		return;
		
	int newrx = ((x + (newdims.x) / 2) % newdims.x);
	int newry = ((blockIdx.y + (newdims.y) / 2) % newdims.y);
	int newrz = ((blockIdx.z + (newdims.z) / 2) % newdims.z);
	
	int oldrx =  newrx + (olddims.x - newdims.x - ((olddims.x & 1 - (newdims.x & 1)) % 2)) / 2;
	int oldry =  newry + (olddims.y - newdims.y - ((olddims.y & 1 - (newdims.y & 1)) % 2)) / 2;
	int oldrz =  newrz + (olddims.z - newdims.z - ((olddims.z & 1 - (newdims.z & 1)) % 2)) / 2;
	
	if(oldrx >= 0 && oldrx < olddims.x && oldry >= 0 && oldry < olddims.y && oldrz >= 0 && oldrz < olddims.z)
	{
		int oldx = ((oldrx + (olddims.x + 1) / 2) % olddims.x);
		int oldy = ((oldry + (olddims.y + 1) / 2) % olddims.y);
		int oldz = ((oldrz + (olddims.z + 1) / 2) % olddims.z);

		d_output[(blockIdx.z * newdims.y + blockIdx.y) * newdims.x + x] = d_input[(oldz * olddims.y + oldy) * olddims.x + oldx];
	}
	else
		d_output[(blockIdx.z * newdims.y + blockIdx.y) * newdims.x + x] = toTcomplex((tfloat)0, (tfloat)0);
}