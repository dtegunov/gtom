#include "Prerequisites.cuh"
#include "Generics.cuh"
#include "Helper.cuh"
#include "ImageManipulation.cuh"

#define MonoTpB 192


////////////////////////////
//CUDA kernel declarations//
////////////////////////////

__global__ void NormPhaseKernel(tfloat* d_input, tfloat* d_output, imgstats5* d_imagestats, size_t elements);
__global__ void NormStdDevKernel(tfloat* d_input, tfloat* d_output, imgstats5* d_imagestats, size_t elements, tfloat stddevmultiple);
__global__ void NormMeanStdDevKernel(tfloat* d_input, tfloat* d_output, imgstats5* d_imagestats, size_t elements);
__global__ void NormCustomScfKernel(tfloat* d_input, tfloat* d_output, imgstats5* d_imagestats, size_t elements, tfloat scf);

template<bool outputmu> __global__ void NormMeanStdDevMonoKernel(tfloat* d_input, tfloat* d_output, tfloat2* d_mu, size_t elements);
template<bool outputmu> __global__ void NormMeanStdDevMonoMaskedKernel(tfloat* d_input, tfloat* d_output, tfloat2* d_mu, tfloat* d_mask, size_t elements);


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

	free(h_imagestats);
	cudaFree(d_imagestats);
}
template void d_Norm<tfloat>(tfloat* d_input, tfloat* d_output, size_t elements, tfloat* d_mask, T_NORM_MODE mode, tfloat stddev, int batch);
template void d_Norm<int>(tfloat* d_input, tfloat* d_output, size_t elements, int* d_mask, T_NORM_MODE mode, tfloat stddev, int batch);
template void d_Norm<char>(tfloat* d_input, tfloat* d_output, size_t elements, char* d_mask, T_NORM_MODE mode, tfloat stddev, int batch);
template void d_Norm<bool>(tfloat* d_input, tfloat* d_output, size_t elements, bool* d_mask, T_NORM_MODE mode, tfloat stddev, int batch);

void d_NormMonolithic(tfloat* d_input, tfloat* d_output, size_t elements, T_NORM_MODE mode, int batch)
{
	for (int b = 0; b < batch; b += 32768)
	{
		dim3 grid = dim3(min(batch - b, 32768));
		NormMeanStdDevMonoKernel<false> << <grid, MonoTpB >> > (d_input + elements * b, d_output + elements * b, NULL, elements);
	}
}

void d_NormMonolithic(tfloat* d_input, tfloat* d_output, tfloat2* d_mu, size_t elements, T_NORM_MODE mode, int batch)
{
	for (int b = 0; b < batch; b += 32768)
	{
		dim3 grid = dim3(min(batch - b, 32768));
		NormMeanStdDevMonoKernel<true> << <grid, MonoTpB >> > (d_input + elements * b, d_output + elements * b, d_mu + b, elements);
	}
}

void d_NormMonolithic(tfloat* d_input, tfloat* d_output, size_t elements, tfloat* d_mask, T_NORM_MODE mode, int batch)
{
	if (d_mask != NULL)
		for (int b = 0; b < batch; b += 32768)
		{
			dim3 grid = dim3(min(batch - b, 32768));
			NormMeanStdDevMonoMaskedKernel<false> << <grid, MonoTpB >> > (d_input + elements * b, d_output + elements * b, NULL, d_mask + elements * b, elements);
		}
	else
		d_NormMonolithic(d_input, d_output, elements, mode, batch);
}

void d_NormMonolithic(tfloat* d_input, tfloat* d_output, tfloat2* d_mu, size_t elements, tfloat* d_mask, T_NORM_MODE mode, int batch)
{
	if (d_mask != NULL)
		for (int b = 0; b < batch; b += 32768)
		{
			dim3 grid = dim3(min(batch - b, 32768));
			NormMeanStdDevMonoMaskedKernel<true> << <grid, MonoTpB >> > (d_input + elements * b, d_output + elements * b, d_mu + b, d_mask + elements * b, elements);
		}
	else
		d_NormMonolithic(d_input, d_output, d_mu, elements, mode, batch);
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

template<bool outputmu> __global__ void NormMeanStdDevMonoKernel(tfloat* d_input, tfloat* d_output, tfloat2* d_mu, size_t elements)
{
	__shared__ double s_sums1[MonoTpB];
	__shared__ double s_sums2[MonoTpB];
	__shared__ double s_mean, s_stddev;

	d_input += elements * blockIdx.x;
	d_output += elements * blockIdx.x;

	double sum1 = 0.0, sum2 = 0.0;

	for (int i = threadIdx.x; i < elements; i += blockDim.x)
	{
		double val = d_input[i];
		sum1 += val;
		sum2 += val * val;
	}
	s_sums1[threadIdx.x] = sum1;
	s_sums2[threadIdx.x] = sum2;
	__syncthreads();

	if (threadIdx.x == 0)
	{
		for (int i = 1; i < MonoTpB; i++)
		{
			sum1 += s_sums1[i];
			sum2 += s_sums2[i];
		}

		s_mean = sum1 / (double)elements;
		s_stddev = sqrt(((double)elements * sum2 - (sum1 * sum1))) / (double)elements;
	}
	__syncthreads();
	
	double mean = s_mean;
	double stddev = s_stddev;

	for (int i = threadIdx.x; i < elements; i += blockDim.x)
		d_output[i] = (d_input[i] - mean) / stddev;

	if (outputmu && threadIdx.x == 0)
		d_mu[blockIdx.x] = tfloat2(mean, stddev);
}

template<bool outputmu> __global__ void NormMeanStdDevMonoMaskedKernel(tfloat* d_input, tfloat* d_output, tfloat2* d_mu, tfloat* d_mask, size_t elements)
{
	__shared__ double s_sums1[MonoTpB];
	__shared__ double s_sums2[MonoTpB];
	__shared__ double s_samples[MonoTpB];
	__shared__ double s_mean, s_stddev;

	d_input += elements * blockIdx.x;
	d_output += elements * blockIdx.x;
	d_mask += elements * blockIdx.x;

	double sum1 = 0.0, sum2 = 0.0, samples = 0.0;

	for (int i = threadIdx.x; i < elements; i += blockDim.x)
	{
		double val = d_input[i];
		double mask = d_mask[i];
		val *= mask;
		sum1 += val;
		sum2 += val * val;
		samples += mask;
	}
	s_sums1[threadIdx.x] = sum1;
	s_sums2[threadIdx.x] = sum2;
	s_samples[threadIdx.x] = samples;
	__syncthreads();

	if (threadIdx.x == 0)
	{
		for (int i = 1; i < MonoTpB; i++)
		{
			sum1 += s_sums1[i];
			sum2 += s_sums2[i];
			samples += s_samples[i];
		}

		s_mean = sum1 / samples;
		s_stddev = sqrt((samples * sum2 - (sum1 * sum1))) / samples;
	}
	__syncthreads();

	double mean = s_mean;
	double stddev = s_stddev;

	for (int i = threadIdx.x; i < elements; i += blockDim.x)
		d_output[i] = (d_input[i] - mean) / stddev;

	if (outputmu && threadIdx.x == 0)
		d_mu[blockIdx.x] = tfloat2(mean, stddev);
}