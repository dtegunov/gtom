#include "../Prerequisites.cuh"
#include "../Functions.cuh"
#include "../DeviceFunctions.cuh"
//#include "../CubicSplines/cubicTex.cu"
//#include "../CubicSplines/internal/cubicTexKernels.cu"


////////////////////////////
//CUDA kernel declarations//
////////////////////////////

__global__ void Interpolate1DLinearKernel(tfloat* d_output, double step, int newdim);
__global__ void Interpolate1DCubicKernel(tfloat* d_output, double step, int newdim);


///////////
//Globals//
///////////

texture<tfloat, 1, cudaReadModeElementType> texScaleInput1d;
texture<tfloat, 2, cudaReadModeElementType> texScaleInput2d;
texture<tfloat, 3, cudaReadModeElementType> texScaleInput3d;


////////////////////////////////////////////////////////////////////////////////
//Combines the functionality of TOM's tom_rescale and MATLAB's interp* methods// 
////////////////////////////////////////////////////////////////////////////////

void d_Scale(tfloat* d_input, tfloat* d_output, int3 olddims, int3 newdims, T_INTERP_MODE mode, cufftHandle* planforw, cufftHandle* planback, int batch)
{
	//Both sizes should have an equal number of dimensions
	int ndims = DimensionCount(olddims);
	if(ndims != DimensionCount(newdims))
		throw;

	//All new dimensions must be either bigger than the old or smaller, not mixed
	int biggerdims = 0;
	for(int i = 0; i < ndims; i++)
		if(((int*)&newdims)[i] > ((int*)&olddims)[i])
			biggerdims++;
	if(biggerdims != 0 && biggerdims != ndims)
		throw;

	size_t elementsold = olddims.x * olddims.y * olddims.z;
	size_t elementsoldFFT = (olddims.x / 2 + 1) * olddims.y * olddims.z;
	size_t elementsnew = newdims.x * newdims.y * newdims.z;
	size_t elementsnewFFT = (newdims.x / 2 + 1) * newdims.y * newdims.z;

	if(mode == T_INTERP_MODE::T_INTERP_LINEAR || mode == T_INTERP_MODE::T_INTERP_CUBIC)
	{
		if(ndims == 1)
		{		
			cudaChannelFormatDesc channelDescInput = cudaCreateChannelDesc<tfloat>();
			cudaArray *d_inputArray = 0;
			cudaMallocArray(&d_inputArray, &channelDescInput, olddims.x, 1);

			for(int b = 0; b < batch; b++)
			{
				cudaMemcpyToArray(d_inputArray, 0, 0, d_input + elementsold * b, olddims.x * sizeof(tfloat), cudaMemcpyDeviceToDevice);
				cudaBindTextureToArray(texScaleInput1d, d_inputArray, channelDescInput);
				texScaleInput1d.normalized = false;
				texScaleInput1d.filterMode = cudaFilterModeLinear;

				int TpB = min(256, newdims.x);
				dim3 grid = dim3(min(8192, (newdims.x + TpB - 1) / TpB));
				if(mode == T_INTERP_MODE::T_INTERP_LINEAR)
					Interpolate1DLinearKernel <<<grid, TpB>>> (d_output + elementsnew * b, (double)olddims.x / (double)newdims.x, newdims.x);
				else
					Interpolate1DCubicKernel <<<grid, TpB>>> (d_output + elementsnew * b, (double)olddims.x / (double)newdims.x, newdims.x);

				cudaUnbindTexture(texScaleInput1d);
			}
		}
		else if(ndims == 2)
		{
	
		}
		else if(ndims == 3)
		{
	
		}
		else
			throw;
	}
	else if(mode == T_INTERP_MODE::T_INTERP_FOURIER)
	{
		tcomplex* d_inputFFT;
		cudaMalloc((void**)&d_inputFFT, elementsoldFFT * sizeof(tcomplex));
		tcomplex* d_inputFFT2;
		cudaMalloc((void**)&d_inputFFT2, elementsold * sizeof(tcomplex));
		tcomplex* d_outputFFT;
		cudaMalloc((void**)&d_outputFFT, elementsnew * sizeof(tcomplex));

		tfloat normfactor = (tfloat)newdims.x / (tfloat)olddims.x * (tfloat)newdims.y / (tfloat)olddims.y * (tfloat)newdims.z / (tfloat)olddims.z;

		for (int b = 0; b < batch; b++)
		{
			if(planforw == NULL)
				d_FFTR2C(d_input + elementsold * b, d_inputFFT, ndims, olddims);
			else
				d_FFTR2C(d_input + elementsold * b, d_inputFFT, planforw);

			d_HermitianSymmetryPad(d_inputFFT, d_inputFFT2, olddims);
			
			if(newdims.x > olddims.x)
				d_FFTFullPad(d_inputFFT2, d_outputFFT, olddims, newdims);
			else
				d_FFTFullCrop(d_inputFFT2, d_outputFFT, olddims, newdims);

			if(planback == NULL)
				d_IFFTC2C(d_outputFFT, d_outputFFT, ndims, newdims);
			else
				d_IFFTC2C(d_outputFFT, d_outputFFT, planback, newdims);

			d_Re(d_outputFFT, d_output + elementsnew * b, elementsnew);
		}

		d_MultiplyByScalar(d_output, d_output, elementsnew * batch, normfactor);
		
		cudaFree(d_inputFFT);
		cudaFree(d_inputFFT2);
		cudaFree(d_outputFFT);
	}
}


////////////////
//CUDA kernels//
////////////////

__global__ void Interpolate1DLinearKernel(tfloat* d_output, double step, int newdim)
{
	for(size_t id = blockIdx.x * blockDim.x + threadIdx.x; 
		id < newdim; 
		id += blockDim.x * gridDim.x)
	{
		tfloat x = (tfloat)((double)id * step);
		d_output[id] = tex1D(texScaleInput1d, x);
	}
}

__global__ void Interpolate1DCubicKernel(tfloat* d_output, double step, int newdim)
{
	for(size_t id = blockIdx.x * blockDim.x + threadIdx.x; 
		id < newdim; 
		id += blockDim.x * gridDim.x)
	{
		float x = (float)((double)id * step);
		d_output[id] = cubicTex1D(texScaleInput1d, x);
	}
}