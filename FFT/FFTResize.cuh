#include "..\Prerequisites.cuh"
#include "..\Functions.cuh"

////////////////////////////
//CUDA kernel declarations//
////////////////////////////

__global__ void FFTCropKernel(tcomplex* d_input, tcomplex* d_output, int3 olddimgs, int3 newdims);


////////////////
//Host methods//
////////////////

void FFTCrop(tcomplex* d_input, tcomplex* d_output, int3 olddims, int3 newdims, int batch)
{
	int TpB = min(256, newdims.x);
	dim3 cropFFTDim = dim3((newdims.x + TpB - 1) / TpB, newdims.y, newdims.z);
	FFTCropKernel <<<cropFFTDim, TpB>>> (dev_input, dev_output, originalWidth, originalHeight, desiredWidth, desiredHeight);
	cudaDeviceSynchronize();
}


////////////////
//CUDA kernels//
////////////////

__global__ void FFTCropKernel(tcomplex* d_input, tcomplex* d_output, int3 olddimgs, int3 newdims)
{
	int idX = blockIdx.x * blockDim.x + threadIdx.x;
	if(idX >= newdims.x)
		return;

	/*if(blockIdx.y < desiredHeight / 2)
		output[idY * desiredWidth + idX] = input[idY * originalWidth + idX];
	else
		output[idY * desiredWidth + idX] = input[(idY + originalHeight - desiredHeight) * originalWidth + idX];*/

	return;
}