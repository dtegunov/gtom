#include "Prerequisites.cuh"
#include "CubicInterp.cuh"
#include "DeviceFunctions.cuh"
#include "Helper.cuh"
#include "Transformation.cuh"


////////////////////////////
//CUDA kernel declarations//
////////////////////////////

__global__ void Cart2PolarLinearKernel(tfloat* d_output, int2 polardims, tfloat radius);
__global__ void Cart2PolarCubicKernel(tfloat* d_output, int2 polardims, tfloat radius);
__global__ void CartAtlas2PolarLinearKernel(tfloat* d_output, tfloat2* d_offsets, int2 polardims, tfloat radius);
__global__ void CartAtlas2PolarCubicKernel(tfloat* d_output, tfloat2* d_offsets, int2 polardims, tfloat radius);


///////////
//Globals//
///////////

texture<tfloat, 2, cudaReadModeElementType> texCoordinatesInput2d;

/////////////////////////////////////////////
//Equivalent of TOM's tom_cart2polar method//
/////////////////////////////////////////////

void d_Cart2Polar(tfloat* d_input, tfloat* d_output, int2 dims, T_INTERP_MODE interpolation, int batch)
{
	int2 polardims = GetCart2PolarSize(dims);

	texCoordinatesInput2d.normalized = false;
	texCoordinatesInput2d.filterMode = cudaFilterModeLinear;

	size_t elements = dims.x * dims.y;
	size_t polarelements = polardims.x * polardims.y;

	tfloat* d_pitched = NULL;
	int pitchedwidth = dims.x * sizeof(tfloat);
	//if((dims.x * sizeof(tfloat)) % 32 != 0)
		d_pitched = (tfloat*)CudaMallocAligned2D(dims.x * sizeof(tfloat), dims.y, &pitchedwidth);

	for (int b = 0; b < batch; b++)
	{
		cudaChannelFormatDesc desc = cudaCreateChannelDesc<tfloat>();
		tfloat* d_offsetinput = d_input + elements * b;
		//if(d_pitched != NULL)
		{
			for (int y = 0; y < dims.y; y++)
				cudaMemcpy((char*)d_pitched + y * pitchedwidth, 
							d_offsetinput + y * dims.x, 
							dims.x * sizeof(tfloat), 
							cudaMemcpyDeviceToDevice);
			d_offsetinput = d_pitched;
		}
			
		if(interpolation == T_INTERP_CUBIC)
			d_CubicBSplinePrefilter2D(d_offsetinput, pitchedwidth, dims);

		cudaBindTexture2D(NULL, 
							texCoordinatesInput2d, 
							d_offsetinput, 
							desc, 
							dims.x, 
							dims.y, 
							pitchedwidth);

		size_t TpB = min(256, polardims.y);
		dim3 grid = dim3((int)((polardims.y + TpB - 1) / TpB), polardims.x);

		if(interpolation == T_INTERP_LINEAR)
			Cart2PolarLinearKernel <<<grid, (uint)TpB>>> (d_output + polarelements * b, polardims, (tfloat)max(dims.x, dims.y) / (tfloat)2);
		else if(interpolation == T_INTERP_CUBIC)
			Cart2PolarCubicKernel <<<grid, (uint)TpB>>> (d_output + polarelements * b, polardims, (tfloat)max(dims.x, dims.y) / (tfloat)2);

		cudaUnbindTexture(texCoordinatesInput2d);
	}

	if(d_pitched != NULL)
		cudaFree(d_pitched);
}


void d_CartAtlas2Polar(tfloat* d_input, tfloat* d_output, tfloat2* d_offsets, int2 atlasdims, int2 dims, T_INTERP_MODE interpolation, int batch)
{
	int2 polardims = GetCart2PolarSize(dims);

	texCoordinatesInput2d.normalized = false;
	texCoordinatesInput2d.filterMode = cudaFilterModeLinear;

	size_t elements = dims.x * dims.y;
	size_t polarelements = polardims.x * polardims.y;

	tfloat* d_pitched = NULL;
	int pitchedwidth = atlasdims.x * sizeof(tfloat);
	//if((dims.x * sizeof(tfloat)) % 32 != 0)
		d_pitched = (tfloat*)CudaMallocAligned2D(atlasdims.x * sizeof(tfloat), atlasdims.y, &pitchedwidth);

	cudaChannelFormatDesc desc = cudaCreateChannelDesc<tfloat>();
	tfloat* d_offsetinput = d_input;
	//if(d_pitched != NULL)
	{
		for (int y = 0; y < atlasdims.y; y++)
			cudaMemcpy((char*)d_pitched + y * pitchedwidth, 
						d_offsetinput + y * atlasdims.x, 
						atlasdims.x * sizeof(tfloat), 
						cudaMemcpyDeviceToDevice);
		d_offsetinput = d_pitched;
	}
			
	if(interpolation == T_INTERP_CUBIC)
		d_CubicBSplinePrefilter2D(d_offsetinput, pitchedwidth, atlasdims);

	cudaBindTexture2D(NULL, 
						texCoordinatesInput2d, 
						d_offsetinput, 
						desc, 
						atlasdims.x, 
						atlasdims.y, 
						pitchedwidth);

	size_t TpB = min(256, polardims.y);
	dim3 grid = dim3((int)((polardims.y + TpB - 1) / TpB), polardims.x, batch);

	if(interpolation == T_INTERP_LINEAR)
		CartAtlas2PolarLinearKernel <<<grid, (uint)TpB>>> (d_output, d_offsets, polardims, (tfloat)max(dims.x, dims.y) / (tfloat)2);
	else if(interpolation == T_INTERP_CUBIC)
		CartAtlas2PolarCubicKernel <<<grid, (uint)TpB>>> (d_output, d_offsets, polardims, (tfloat)max(dims.x, dims.y) / (tfloat)2);

	cudaUnbindTexture(texCoordinatesInput2d);

	if(d_pitched != NULL)
		cudaFree(d_pitched);
}

int2 GetCart2PolarSize(int2 dims)
{
	int2 polardims;
	polardims.x = max(dims.x, dims.y) / 2;		//radial
	polardims.y = max(dims.x, dims.y) * 2;		//angular

	return polardims;
}


////////////////
//CUDA kernels//
////////////////

__global__ void Cart2PolarLinearKernel(tfloat* d_output, int2 polardims, tfloat radius)
{
	int idy = blockIdx.x * blockDim.x + threadIdx.x;
	if(idy >= polardims.y)
		return;
	int idx = blockIdx.y;

	tfloat r = (tfloat)idx;
	tfloat phi = (tfloat)(idy) * PI2 / (tfloat)polardims.y;

	d_output[idy * polardims.x + idx] = tex2D(texCoordinatesInput2d, 
											  cos(phi) * r + radius + (tfloat)0.5, 
											  sin(phi) * r + radius + (tfloat)0.5);
}

__global__ void Cart2PolarCubicKernel(tfloat* d_output, int2 polardims, tfloat radius)
{
	int idy = blockIdx.x * blockDim.x + threadIdx.x;
	if(idy >= polardims.y)
		return;
	int idx = blockIdx.y;

	tfloat r = (tfloat)idx;
	tfloat phi = (tfloat)(idy) * PI2 / (tfloat)polardims.y;

	d_output[idy * polardims.x + idx] = cubicTex2D(texCoordinatesInput2d, 
												  cos(phi) * r + radius + (tfloat)0.5, 
												  sin(phi) * r + radius + (tfloat)0.5);
}

__global__ void CartAtlas2PolarLinearKernel(tfloat* d_output, tfloat2* d_offsets, int2 polardims, tfloat radius)
{
	int idy = blockIdx.x * blockDim.x + threadIdx.x;
	if(idy >= polardims.y)
		return;
	int idx = blockIdx.y;

	tfloat r = (tfloat)idx;
	tfloat phi = (tfloat)(idy) * PI2 / (tfloat)polardims.y;

	d_output[blockIdx.z * polardims.x * polardims.y + idy * polardims.x + idx] = tex2D(texCoordinatesInput2d, 
																					  d_offsets[blockIdx.z].x + cos(phi) * r + radius + (tfloat)0.5, 
																					  d_offsets[blockIdx.z].y + sin(phi) * r + radius + (tfloat)0.5);
}

__global__ void CartAtlas2PolarCubicKernel(tfloat* d_output, tfloat2* d_offsets, int2 polardims, tfloat radius)
{
	int idy = blockIdx.x * blockDim.x + threadIdx.x;
	if(idy >= polardims.y)
		return;
	int idx = blockIdx.y;

	tfloat r = (tfloat)idx;
	tfloat phi = (tfloat)(idy) * PI2 / (tfloat)polardims.y;

	d_output[blockIdx.z * polardims.x * polardims.y + idy * polardims.x + idx] = cubicTex2D(texCoordinatesInput2d, 
																							d_offsets[blockIdx.z].x + cos(phi) * r + radius + (tfloat)0.5, 
																							d_offsets[blockIdx.z].y + sin(phi) * r + radius + (tfloat)0.5);
}