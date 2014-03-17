#include "../Prerequisites.cuh"
#include "../Functions.cuh"

#define MonoTpB 160


////////////////////////////
//CUDA kernel declarations//
////////////////////////////

__global__ void NormPhaseKernel(tfloat* d_input, tfloat* d_output, imgstats5* d_imagestats, size_t elements);
__global__ void NormStdDevKernel(tfloat* d_input, tfloat* d_output, imgstats5* d_imagestats, size_t elements, tfloat stddevmultiple);
__global__ void NormMeanStdDevKernel(tfloat* d_input, tfloat* d_output, imgstats5* d_imagestats, size_t elements);
__global__ void NormCustomScfKernel(tfloat* d_input, tfloat* d_output, imgstats5* d_imagestats, size_t elements, tfloat scf);

__global__ void NormMeanStdDevMonoKernel(tfloat* d_input, tfloat* d_output, int elements, int batch);


///////////////////////////////////////
//Equivalent of TOM's tom_norm method//
///////////////////////////////////////

template <class Tmask> void d_Norm(tfloat* d_input, tfloat* d_output, size_t elements, Tmask* d_mask, T_NORM_MODE mode, tfloat scf, int batch)
{
	for (int b = 0; b < batch; b++)
	{
		imgstats5* d_imagestats;
		cudaMalloc((void**)&d_imagestats, batch * sizeof(imgstats5));
		d_Dev(d_input + elements * b, d_imagestats, elements, d_mask, batch);

		imgstats5* h_imagestats = (imgstats5*)MallocFromDeviceArray(d_imagestats, batch * sizeof(imgstats5));

		size_t TpB = min(192, NextMultipleOf(elements, 32));
		size_t totalblocks = min((elements + TpB - 1) / TpB, 32768);
		dim3 grid = dim3((uint)totalblocks, batch);

		if(mode == T_NORM_MODE::T_NORM_PHASE)
			NormPhaseKernel <<<grid, (uint)TpB>>> (d_input + elements * b, d_output + elements * b, d_imagestats, elements);
		else if(mode == T_NORM_MODE::T_NORM_STD1)
			NormStdDevKernel <<<grid, (uint)TpB>>> (d_input + elements * b, d_output + elements * b, d_imagestats, elements, (tfloat)1);
		else if(mode == T_NORM_MODE::T_NORM_STD2)
			NormStdDevKernel <<<grid, (uint)TpB>>> (d_input + elements * b, d_output + elements * b, d_imagestats, elements, (tfloat)2);
		else if(mode == T_NORM_MODE::T_NORM_STD3)
			NormStdDevKernel <<<grid, (uint)TpB>>> (d_input + elements * b, d_output + elements * b, d_imagestats, elements, (tfloat)3);
		else if(mode == T_NORM_MODE::T_NORM_MEAN01STD)
			NormMeanStdDevKernel <<<grid, (uint)TpB>>> (d_input + elements * b, d_output + elements * b, d_imagestats, elements);
		else if(mode == T_NORM_MODE::T_NORM_OSCAR)
		{
			NormStdDevKernel <<<grid, (uint)TpB>>> (d_input + elements * b, d_output + elements * b, d_imagestats, elements, (tfloat)3);
			d_AddScalar(d_output + elements * b, d_output + elements * b, elements * batch, (tfloat)100);
			d_Dev(d_output + elements * b, d_imagestats, elements, d_mask, batch);
			NormPhaseKernel <<<grid, (uint)TpB>>> (d_output + elements * b, d_output + elements * b, d_imagestats, elements);
		}
		else if(mode == T_NORM_MODE::T_NORM_CUSTOM)
			NormCustomScfKernel <<<grid, (uint)TpB>>> (d_input + elements * b, d_output + elements * b, d_imagestats, elements, scf);

		free(h_imagestats);
		cudaFree(d_imagestats);
	}
}
template void d_Norm<tfloat>(tfloat* d_input, tfloat* d_output, size_t elements, tfloat* d_mask, T_NORM_MODE mode, tfloat stddev, int batch);
template void d_Norm<int>(tfloat* d_input, tfloat* d_output, size_t elements, int* d_mask, T_NORM_MODE mode, tfloat stddev, int batch);
template void d_Norm<char>(tfloat* d_input, tfloat* d_output, size_t elements, char* d_mask, T_NORM_MODE mode, tfloat stddev, int batch);
template void d_Norm<bool>(tfloat* d_input, tfloat* d_output, size_t elements, bool* d_mask, T_NORM_MODE mode, tfloat stddev, int batch);

void d_NormMonolithic(tfloat* d_input, tfloat* d_output, size_t elements, T_NORM_MODE mode, int batch)
{
	size_t TpB = MonoTpB;
	size_t totalblocks = min(batch, 32768);
	dim3 grid = dim3((uint)totalblocks);

	NormMeanStdDevMonoKernel <<<grid, TpB>>> (d_input, d_output, elements, batch);
}


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

__global__ void NormMeanStdDevMonoKernel(tfloat* d_input, tfloat* d_output, int elements, int batch)
{
	__shared__ tfloat means[MonoTpB];
	__shared__ tfloat mean;
	__shared__ tfloat stddev;

	for (int b = blockIdx.x; b < batch; b += gridDim.x)
	{
		tfloat localmean, localstddev;
		means[threadIdx.x] = (tfloat)0;

		tfloat* offsetinput = d_input + (size_t)elements * (size_t)b;
		tfloat* offsetoutput = d_output + (size_t)elements * (size_t)b;

		for (int i = threadIdx.x; i < elements; i += MonoTpB)
			means[threadIdx.x] += offsetinput[i];

		__syncthreads();

		if(threadIdx.x == 0)
		{
			mean = (tfloat)0;
			for (int i = 0; i < MonoTpB; i++)
				mean += means[i];
			mean /= (tfloat)elements;
		}
		__syncthreads();

		localmean = mean;
		means[threadIdx.x] = (tfloat)0;

		tfloat diff;
		for (int i = threadIdx.x; i < elements; i += MonoTpB)
		{
			diff = offsetinput[i] - localmean;
			means[threadIdx.x] += diff * diff;
		}
		__syncthreads();

		if(threadIdx.x == 0)
		{
			stddev = (tfloat)0;
			for (int i = 0; i < MonoTpB; i++)
				stddev += means[i];
			stddev = sqrt(stddev / (tfloat)(elements - 1));
		}
		__syncthreads();

		localstddev = stddev;
		for (int i = threadIdx.x; i < elements; i += MonoTpB)
			offsetoutput[i] = (offsetinput[i] - localmean) / localstddev;
	}
}