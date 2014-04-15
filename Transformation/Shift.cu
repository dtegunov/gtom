#include "../Prerequisites.cuh"
#include "../Functions.cuh"


////////////////////////////
//CUDA kernel declarations//
////////////////////////////

template<int ndims, bool iszerocentered> __global__ void ShiftFourierKernel(tcomplex* d_input, tcomplex* d_output, int3 dims, tfloat3 delta);
__global__ void ShiftIntegerKernel(tfloat* d_input, tfloat* d_output, int3 dims, int3 delta);


////////////////////////////////////////
//Equivalent of TOM's tom_shift method//
////////////////////////////////////////

void d_Shift(tfloat* d_input, tfloat* d_output, int3 dims, tfloat3* delta, cufftHandle* planforw, cufftHandle* planback, tcomplex* d_sharedintermediate, int batch)
{
	tcomplex* d_intermediate = NULL;
	if(d_sharedintermediate == NULL)
		CudaSafeCall(cudaMalloc((void**)&d_intermediate, ElementsFFT(dims) * sizeof(tcomplex)));
	else
		d_intermediate = d_sharedintermediate;

	for (int b = 0; b < batch; b++)
	{
		if(fmod(delta[b].x, (tfloat)1) != (tfloat)0 || fmod(delta[b].y, (tfloat)1) != (tfloat)0 || fmod(delta[b].z, (tfloat)1) != (tfloat)0)
		{
			tfloat3 normdelta = tfloat3(delta[b].x / (tfloat)dims.x, delta[b].y / (tfloat)dims.y, delta[b].z / (tfloat)dims.z);

			if(planforw == NULL)
				d_FFTR2C(d_input + Elements(dims) * b, d_intermediate, DimensionCount(dims), dims);
			else
				d_FFTR2C(d_input + Elements(dims) * b, d_intermediate, planforw);

			int TpB = min(256, NextMultipleOf(dims.x / 2 + 1, 32));
			dim3 grid = dim3(((dims.x / 2 + 1) + TpB - 1) / TpB, dims.y, dims.z);
			if(DimensionCount(dims) == 3)
				ShiftFourierKernel <3, false> <<<grid, TpB>>> (d_intermediate, d_intermediate, dims, normdelta);
			else
				ShiftFourierKernel <2, false> <<<grid, TpB>>> (d_intermediate, d_intermediate, dims, normdelta);
			cudaStreamQuery(0);
			
			if(planback == NULL)
				d_IFFTC2R(d_intermediate, d_output + Elements(dims) * b, DimensionCount(dims), dims);
			else
				d_IFFTC2R(d_intermediate, d_output + Elements(dims) * b, planback, dims);
		}
		else
		{
			int TpB = min(256, NextMultipleOf(dims.x, 32));
			dim3 grid = dim3((dims.x + TpB - 1) / TpB, dims.y, dims.z);
			ShiftIntegerKernel <<<grid, TpB>>> (d_input + Elements(dims) * b, d_input == d_output ? (tfloat*)d_intermediate : (d_output + Elements(dims) * b), dims, toInt3((int)delta[b].x, (int)delta[b].y, (int)delta[b].z));
			cudaStreamQuery(0);

			if(d_input == d_output)
				cudaMemcpy(d_output + Elements(dims) * b, d_intermediate, Elements(dims) * sizeof(tfloat), cudaMemcpyDeviceToDevice);
		}
	}

	if(d_sharedintermediate == NULL)
		cudaFree(d_intermediate);
}

void d_Shift(tcomplex* d_input, tcomplex* d_output, int3 dims, tfloat3* delta, bool iszerocentered, int batch)
{
	for (int b = 0; b < batch; b++)
	{
		tfloat3 normdelta = tfloat3(delta[b].x / (tfloat)dims.x, delta[b].y / (tfloat)dims.y, delta[b].z / (tfloat)dims.z);

		int TpB = min(256, NextMultipleOf(dims.x / 2 + 1, 32));
		dim3 grid = dim3(((dims.x / 2 + 1) + TpB - 1) / TpB, dims.y, dims.z);
		if(!iszerocentered)
		{
			if(DimensionCount(dims) == 3)
				ShiftFourierKernel <3, false> <<<grid, TpB>>> (d_input + ElementsFFT(dims) * b, d_output + ElementsFFT(dims) * b, dims, normdelta);
			else
				ShiftFourierKernel <2, false> <<<grid, TpB>>> (d_input + ElementsFFT(dims) * b, d_output + ElementsFFT(dims) * b, dims, normdelta);
		}
		else
		{
			if(DimensionCount(dims) == 3)
				ShiftFourierKernel <3, false> <<<grid, TpB>>> (d_input + ElementsFFT(dims) * b, d_output + ElementsFFT(dims) * b, dims, normdelta);
			else
				ShiftFourierKernel <2, false> <<<grid, TpB>>> (d_input + ElementsFFT(dims) * b, d_output + ElementsFFT(dims) * b, dims, normdelta);
		}
		cudaStreamQuery(0);
	}
}


////////////////
//CUDA kernels//
////////////////

template<int ndims, bool iszerocentered> __global__ void ShiftFourierKernel(tcomplex* d_input, tcomplex* d_output, int3 dims, tfloat3 delta)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if(x >= dims.x / 2 + 1)
		return;
	int y, z;

	if(!iszerocentered)
	{
		if(x == dims.x / 2)
			x = (-x);
		y = ((blockIdx.y + ((dims.y + 1) / 2)) % dims.y) - (dims.y / 2);
		z = ((blockIdx.z + ((dims.z + 1) / 2)) % dims.z) - (dims.z / 2);
	}
	else
	{
		x = dims.x / 2 - x;
		//if(x == dims.x / 2)
			//x = (-x);
		y = blockIdx.y - (dims.y / 2);
		z = blockIdx.z - (dims.z / 2);
	}

	tfloat factorx = delta.x * (tfloat)x * (tfloat)PI2;
	tcomplex multx = { cos(factorx), sin(-factorx) };
	if(dims.x % 2 == 0 && blockIdx.x * blockDim.x + threadIdx.x == dims.x / 2)
		multx.y = (tfloat)0;

	tfloat factory = delta.y * (tfloat)y * (tfloat)PI2;
	tcomplex multy = { cos(factory), sin(-factory) };
	if(dims.y % 2 == 0 && blockIdx.y == dims.y / 2)
		multy.y = (tfloat)0;

	tcomplex multiplicator = cmul(multx, multy);

	if(ndims > 2)
	{
		tfloat factorz = delta.z * (tfloat)z * (tfloat)PI2;
		tcomplex multz = { cos(factorz), sin(-factorz) };
		if(dims.z % 2 == 0 && blockIdx.z == dims.z / 2)
			multz.y = (tfloat)0;
		multiplicator = cmul(multiplicator, multz);
	}	

	size_t id = (blockIdx.z * dims.y + blockIdx.y) * (dims.x / 2 + 1) + blockIdx.x * blockDim.x + threadIdx.x;
	d_output[id] = cmul(d_input[id], multiplicator);
}

__global__ void ShiftIntegerKernel(tfloat* d_input, tfloat* d_output, int3 dims, int3 delta)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if(x >= dims.x)
		return;

	d_output[(((dims.z + (int)blockIdx.z + delta.z) % dims.z) * dims.y + ((dims.y + (int)blockIdx.y + delta.y) % dims.y)) * dims.x + ((dims.x + x + delta.x) % dims.x)] = d_input[(blockIdx.z * dims.y + blockIdx.y) * dims.x + x];
}