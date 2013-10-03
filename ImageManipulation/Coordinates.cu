#include "..\Prerequisites.cuh"
#include "..\Functions.cuh"


////////////////////////////
//CUDA kernel declarations//
////////////////////////////

__global__ void Cart2PolarKernel(tfloat* d_output, int2 polardims, tfloat radius);


///////////
//Globals//
///////////

texture<tfloat, 2> texInput2d;

/////////////////////////////////////////////
//Equivalent of TOM's tom_cart2polar method//
/////////////////////////////////////////////

void d_Cart2Polar(tfloat* d_input, tfloat* d_output, int2 dims, int batch)
{
	int2 polardims = GetCart2PolarSize(dims);

	texInput2d.normalized = false;
	texInput2d.filterMode = cudaFilterModeLinear;

	size_t elements = dims.x * dims.y;
	size_t polarelements = polardims.x * polardims.y;

	for (int b = 0; b < batch; b++)
	{
		cudaChannelFormatDesc desc = cudaCreateChannelDesc<tfloat>();
		cudaBindTexture2D(NULL, 
							texInput2d, 
							d_input + elements * b, 
							desc, 
							dims.x, 
							dims.y, 
							dims.x * sizeof(tfloat));

		size_t TpB = min(256, polardims.y);
		dim3 grid = dim3((polardims.y + TpB - 1) / TpB, polardims.x);

		Cart2PolarKernel <<<grid, (uint)TpB>>> (d_output + polarelements * b, polardims, (tfloat)max(dims.x, dims.y) / (tfloat)2);

		cudaUnbindTexture(texInput2d);
	}
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

__global__ void Cart2PolarKernel(tfloat* d_output, int2 polardims, tfloat radius)
{
	int idy = blockIdx.x * blockDim.x + threadIdx.x;
	if(idy >= polardims.y)
		return;
	int idx = blockIdx.y;

	tfloat r = (tfloat)idx;
	tfloat phi = (tfloat)(idy) * PI2 / (tfloat)polardims.y;

	d_output[idy * polardims.x + idx] = tex2D(texInput2d, 
											  cos(phi) * r + radius + (tfloat)0.5, 
											  sin(phi) * r + radius + (tfloat)0.5);
}