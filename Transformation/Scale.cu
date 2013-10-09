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

void d_Scale(tfloat* d_input, tfloat* d_output, int3 olddims, int3 newdims, T_INTERP_MODE mode, int batch)
{
	int ndims = DimensionCount(olddims);
	if(ndims != DimensionCount(newdims))
		throw;

	if(ndims == 1)
	{
		if(mode == T_INTERP_MODE::T_INTERP_LINEAR || mode == T_INTERP_MODE::T_INTERP_CUBIC)
		{
			//d_CubicBSplinePrefilter2D(d_input, olddims.x * sizeof(tfloat), olddims);

			cudaChannelFormatDesc channelDescInput = cudaCreateChannelDesc<tfloat>();
			cudaArray *d_inputArray = 0;
			cudaMallocArray(&d_inputArray, &channelDescInput, olddims.x, 1);
			cudaMemcpyToArray(d_inputArray, 0, 0, d_input, olddims.x * sizeof(tfloat), cudaMemcpyDeviceToDevice);
			//cudaMemcpy2DToArray(d_inputArray, 0, 0, d_input, 0, olddims.x * sizeof(tfloat), 1, cudaMemcpyDeviceToDevice);
			cudaBindTextureToArray(texScaleInput1d, d_inputArray, channelDescInput);
			texScaleInput1d.normalized = false;
			texScaleInput1d.filterMode = cudaFilterModeLinear;

			int TpB = min(256, newdims.x);
			dim3 grid = dim3(min(8192, (newdims.x + TpB - 1) / TpB));
			if(mode == T_INTERP_MODE::T_INTERP_LINEAR)
				Interpolate1DLinearKernel <<<grid, TpB>>> (d_output, (double)olddims.x / (double)newdims.x, newdims.x);
			else
				Interpolate1DCubicKernel <<<grid, TpB>>> (d_output, (double)olddims.x / (double)newdims.x, newdims.x);

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