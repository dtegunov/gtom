#include "..\Prerequisites.cuh"
#include "..\Functions.cuh"


////////////////////////////
//CUDA kernel declarations//
////////////////////////////

__global__ void NormPhaseKernel(tfloat* d_input, tfloat* d_output, imgstats5* d_imagestats, size_t elements);
__global__ void NormStdDevKernel(tfloat* d_input, tfloat* d_output, imgstats5* d_imagestats, size_t elements, tfloat stddevmultiple);
__global__ void NormMeanStdDevKernel(tfloat* d_input, tfloat* d_output, imgstats5* d_imagestats, size_t elements);
__global__ void NormCustomScfKernel(tfloat* d_input, tfloat* d_output, imgstats5* d_imagestats, size_t elements, tfloat scf);


///////////////////////////////////////
//Equivalent of TOM's tom_norm method//
///////////////////////////////////////

template <class Tmask> void d_Norm(tfloat* d_input, tfloat* d_output, size_t elements, Tmask* d_mask, T_NORM_MODE mode, tfloat scf, int batch)
{
	imgstats5* d_imagestats;
	cudaMalloc((void**)&d_imagestats, batch * sizeof(imgstats5));
	d_Dev(d_input, d_imagestats, elements, d_mask, batch);

	imgstats5* h_imagestats = (imgstats5*)MallocFromDeviceArray(d_imagestats, batch * sizeof(imgstats5));

	size_t TpB = min(192, NextMultipleOf(elements, 32));
	size_t totalblocks = min((elements + TpB - 1) / TpB, 32768);
	dim3 grid = dim3((uint)totalblocks, batch);

	if(mode == T_NORM_MODE::T_NORM_PHASE)
		NormPhaseKernel <<<grid, (uint)TpB>>> (d_input, d_output, d_imagestats, elements);
	else if(mode == T_NORM_MODE::T_NORM_STD1)
		NormStdDevKernel <<<grid, (uint)TpB>>> (d_input, d_output, d_imagestats, elements, (tfloat)1);
	else if(mode == T_NORM_MODE::T_NORM_STD2)
		NormStdDevKernel <<<grid, (uint)TpB>>> (d_input, d_output, d_imagestats, elements, (tfloat)2);
	else if(mode == T_NORM_MODE::T_NORM_STD3)
		NormStdDevKernel <<<grid, (uint)TpB>>> (d_input, d_output, d_imagestats, elements, (tfloat)3);
	else if(mode == T_NORM_MODE::T_NORM_MEAN01STD)
		NormMeanStdDevKernel <<<grid, (uint)TpB>>> (d_input, d_output, d_imagestats, elements);
	else if(mode == T_NORM_MODE::T_NORM_OSCAR)
	{
		NormStdDevKernel <<<grid, (uint)TpB>>> (d_input, d_output, d_imagestats, elements, (tfloat)3);
		d_AddScalar(d_output, d_output, elements * batch, (tfloat)100);
		d_Dev(d_output, d_imagestats, elements, d_mask, batch);
		NormPhaseKernel <<<grid, (uint)TpB>>> (d_output, d_output, d_imagestats, elements);
	}
	else if(mode == T_NORM_MODE::T_NORM_CUSTOM)
		NormCustomScfKernel <<<grid, (uint)TpB>>> (d_input, d_output, d_imagestats, elements, scf);

	tfloat* h_output = (tfloat*)MallocFromDeviceArray(d_output, elements * sizeof(tfloat));
}
template void d_Norm<tfloat>(tfloat* d_input, tfloat* d_output, size_t elements, tfloat* d_mask, T_NORM_MODE mode, tfloat stddev, int batch);
template void d_Norm<int>(tfloat* d_input, tfloat* d_output, size_t elements, int* d_mask, T_NORM_MODE mode, tfloat stddev, int batch);
template void d_Norm<char>(tfloat* d_input, tfloat* d_output, size_t elements, char* d_mask, T_NORM_MODE mode, tfloat stddev, int batch);
template void d_Norm<bool>(tfloat* d_input, tfloat* d_output, size_t elements, bool* d_mask, T_NORM_MODE mode, tfloat stddev, int batch);


////////////////
//CUDA kernels//
////////////////

__global__ void NormPhaseKernel(tfloat* d_input, tfloat* d_output, imgstats5* d_imagestats, size_t elements)
{
	__shared__ tfloat mean;
	if(threadIdx.x == 0)
		mean = d_imagestats[blockIdx.y].mean;
	__syncthreads();

	size_t offset = elements * blockIdx.y;
	for(size_t id = blockIdx.x * blockDim.x + threadIdx.x; 
		id < elements; 
		id += blockDim.x * gridDim.x)
		d_output[id + offset] = (d_input[id + offset] - mean) / (mean + (tfloat)0.000000000001);
}

__global__ void NormStdDevKernel(tfloat* d_input, tfloat* d_output, imgstats5* d_imagestats, size_t elements, tfloat stddevmultiple)
{
	__shared__ tfloat mean, stddev;
	if(threadIdx.x == 0)
		stddev = d_imagestats[blockIdx.y].stddev * stddevmultiple;
	else if(threadIdx.x == 1)
		mean = d_imagestats[blockIdx.y].mean;
	__syncthreads();

	size_t offset = elements * blockIdx.y;
	tfloat upper = mean + stddev, lower = mean - stddev;
	for(size_t id = blockIdx.x * blockDim.x + threadIdx.x; 
		id < elements; 
		id += blockDim.x * gridDim.x)
		d_output[id + offset] = max(min(d_input[id + offset], upper), lower) - mean;
}

__global__ void NormMeanStdDevKernel(tfloat* d_input, tfloat* d_output, imgstats5* d_imagestats, size_t elements)
{
	__shared__ tfloat mean, stddev;
	if(threadIdx.x == 0)
		stddev = d_imagestats[blockIdx.y].stddev;
	else if(threadIdx.x == 1)
		mean = d_imagestats[blockIdx.y].mean;
	__syncthreads();

	size_t offset = elements * blockIdx.y;
	if(stddev == (tfloat)0)
		for(size_t id = blockIdx.x * blockDim.x + threadIdx.x; 
			id < elements; 
			id += blockDim.x * gridDim.x)
			d_output[id + offset] = d_input[id + offset] - mean;
	else
		for(size_t id = blockIdx.x * blockDim.x + threadIdx.x; 
			id < elements; 
			id += blockDim.x * gridDim.x)
			d_output[id + offset] = (d_input[id + offset] - mean) / stddev;
}

__global__ void NormCustomScfKernel(tfloat* d_input, tfloat* d_output, imgstats5* d_imagestats, size_t elements, tfloat scf)
{
	__shared__ imgstats5 stats;
	if(threadIdx.x == 0)
		stats = d_imagestats[blockIdx.y];
	__syncthreads();

	size_t offset = elements * blockIdx.y;
	if(stats.stddev != (tfloat)0 && stats.mean != scf)
	{		
		tfloat range = stats.max - stats.min;
		for(size_t id = blockIdx.x * blockDim.x + threadIdx.x; 
			id < elements; 
			id += blockDim.x * gridDim.x)
			d_output[id + offset] = scf * (d_input[id + offset] - stats.min) / range;
	}
	else if(stats.stddev == (tfloat)0 && stats.mean != scf)
		for(size_t id = blockIdx.x * blockDim.x + threadIdx.x; 
			id < elements; 
			id += blockDim.x * gridDim.x)
			d_output[id + offset] = d_input[id + offset] / scf;
	else
		for(size_t id = blockIdx.x * blockDim.x + threadIdx.x; 
			id < elements; 
			id += blockDim.x * gridDim.x)
			d_output[id + offset] = d_input[id + offset];
}