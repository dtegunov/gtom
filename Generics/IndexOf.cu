#include "../Prerequisites.cuh"
#include "../Functions.cuh"


////////////////////////////
//CUDA kernel declarations//
////////////////////////////

__global__ void FirstIndexOfLinearKernel(tfloat* d_input, tfloat* d_output, size_t elements, tfloat value);
__global__ void FirstMinimumLinearKernel(tfloat* d_input, tfloat* d_output, size_t elements);

template<class T> __global__ void BiggerThanKernel(tfloat* d_input, T* d_output, size_t elements, tfloat value);
template<class T> __global__ void SmallerThanKernel(tfloat* d_input, T* d_output, size_t elements, tfloat value);
template<class T> __global__ void IsBetweenKernel(tfloat* d_input, T* d_output, size_t elements, tfloat minval, tfloat maxval);


//////////////////
//First Index Of//
//////////////////

void d_FirstIndexOf(tfloat* d_input, tfloat* d_output, size_t elements, tfloat value, T_INTERP_MODE mode, int batch)
{
	if(mode == T_INTERP_LINEAR)
	{
		int TpB = min(NextMultipleOf(elements, 32), 256);
		dim3 grid = dim3(batch);
		FirstIndexOfLinearKernel <<<grid, TpB>>> (d_input, d_output, elements, value);
	}
	else
		throw;
}

__global__ void FirstIndexOfLinearKernel(tfloat* d_input, tfloat* d_output, size_t elements, tfloat value)
{
	d_input += blockIdx.x * elements;
	
	__shared__ tfloat indices[256];
	__shared__ bool found, anybigger, nan;
	if(threadIdx.x == 0)
	{
		found = false;
		anybigger = false;
		nan = false;
	}
	__syncthreads();

	tfloat index = (tfloat)(elements + 1);
	tfloat current, next;
	for (size_t n = threadIdx.x; n < elements - 1; n += blockDim.x)
	{
		if(found)
			break;

		current = d_input[n];		
		next = d_input[n + 1];
		if(isnan(current) || isnan(next))
		{
			nan = true;
			continue;
		}
		
		if(value < current)
			anybigger = true;

		if((value <= current && value >= next) || (value >= current && value <= next))
		{
			index = (tfloat)n + max(min((value - current) / (next - current + (tfloat)0.00001), 1), 0);
			found = true;
			break;
		}
	}

	indices[threadIdx.x] = index;

	__syncthreads();

	if(threadIdx.x == 0)
	{
		for (int t = 1; t < min(elements, blockDim.x); t++)
			index = min(index, indices[t]);

		if(found)
			d_output[blockIdx.x] = max(index, 1);
		else if(anybigger)
			d_output[blockIdx.x] = (tfloat)elements;
		else if(nan)
			d_output[blockIdx.x] = (tfloat)-1;
		else
			d_output[blockIdx.x] = (tfloat)1;
	}
}

void d_FirstMinimum(tfloat* d_input, tfloat* d_output, size_t elements, T_INTERP_MODE mode, int batch)
{
	if(mode == T_INTERP_LINEAR)
	{
		int TpB = min(NextMultipleOf(elements, 32), 256);
		dim3 grid = dim3(batch);
		FirstMinimumLinearKernel <<<grid, TpB>>> (d_input, d_output, elements);
	}
	else
		throw;
}

__global__ void FirstMinimumLinearKernel(tfloat* d_input, tfloat* d_output, size_t elements)
{
	d_input += blockIdx.x * elements;
	
	__shared__ tfloat indices[256];
	__shared__ bool found, anybigger, nan;
	if(threadIdx.x == 0)
	{
		found = false;
		anybigger = false;
		nan = false;
	}
	__syncthreads();

	tfloat index = (tfloat)(elements + 1);
	tfloat left, right, current;
	for (size_t n = threadIdx.x + 1; n < elements - 1; n += blockDim.x)
	{
		left = d_input[n - 1];		
		current = d_input[n];
		right = d_input[n + 1];

		if(isnan(left) || isnan(current) || isnan(right))
		{
			nan = true;
			continue;
		}
		
		if(current < left && current < right)
		{
			index = (tfloat)n;
			found = true;
			break;
		}
	}

	indices[threadIdx.x] = index;

	__syncthreads();

	if(threadIdx.x == 0)
	{
		for (int t = 1; t < min(elements, blockDim.x); t++)
			index = min(index, indices[t]);

		if(found)
			d_output[blockIdx.x] = max(index, 1);
		else if(anybigger)
			d_output[blockIdx.x] = (tfloat)elements;
		else if(nan)
			d_output[blockIdx.x] = (tfloat)-1;
		else
			d_output[blockIdx.x] = (tfloat)1;
	}
}


//////////////////
//Is Bigger Than//
//////////////////

template<class T> void d_BiggerThan(tfloat* d_input, T* d_output, size_t elements, tfloat value, int batch)
{
	int TpB = min(256, NextMultipleOf(elements, 32));
	dim3 grid = dim3(min((elements + TpB - 1) / TpB, 32768), batch);
	BiggerThanKernel <<<grid, TpB>>> (d_input, d_output, elements, value);
}
template void d_BiggerThan<tfloat>(tfloat* d_input, tfloat* d_output, size_t elements, tfloat value, int batch);
template void d_BiggerThan<char>(tfloat* d_input, char* d_output, size_t elements, tfloat value, int batch);

template<class T> __global__ void BiggerThanKernel(tfloat* d_input, T* d_output, size_t elements, tfloat value)
{
	d_input += elements * blockIdx.y;
	d_output += elements * blockIdx.y;
	
	for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < elements; i += gridDim.x * blockDim.x)
		d_output[i] = d_input[i] > value ? (T)1 : (T)0;
}


///////////////////
//Is Smaller Than//
///////////////////

template<class T> void d_SmallerThan(tfloat* d_input, T* d_output, size_t elements, tfloat value, int batch)
{
	int TpB = min(256, NextMultipleOf(elements, 32));
	dim3 grid = dim3(min((elements + TpB - 1) / TpB, 32768), batch);
	SmallerThanKernel <<<grid, TpB>>> (d_input, d_output, elements, value);
}
template void d_SmallerThan<tfloat>(tfloat* d_input, tfloat* d_output, size_t elements, tfloat value, int batch);
template void d_SmallerThan<char>(tfloat* d_input, char* d_output, size_t elements, tfloat value, int batch);

template<class T> __global__ void SmallerThanKernel(tfloat* d_input, T* d_output, size_t elements, tfloat value)
{
	d_input += elements * blockIdx.y;
	d_output += elements * blockIdx.y;

	for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < elements; i += gridDim.x * blockDim.x)
		d_output[i] = d_input[i] < value ? (T)1 : (T)0;
}


//////////////
//Is Between//
//////////////

template<class T> void d_IsBetween(tfloat* d_input, T* d_output, size_t elements, tfloat minval, tfloat maxval, int batch)
{
	int TpB = min(256, NextMultipleOf(elements, 32));
	dim3 grid = dim3(min((elements + TpB - 1) / TpB, 32768), batch);
	IsBetweenKernel <<<grid, TpB>>> (d_input, d_output, elements, minval, maxval);
}
template void d_IsBetween<tfloat>(tfloat* d_input, tfloat* d_output, size_t elements, tfloat minval, tfloat maxval, int batch);
template void d_IsBetween<char>(tfloat* d_input, char* d_output, size_t elements, tfloat minval, tfloat maxval, int batch);

template<class T> __global__ void IsBetweenKernel(tfloat* d_input, T* d_output, size_t elements, tfloat minval, tfloat maxval)
{
	d_input += elements * blockIdx.y;
	d_output += elements * blockIdx.y;

	for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < elements; i += gridDim.x * blockDim.x)
		d_output[i] = (d_input[i] < maxval && d_input[i] >= minval) ? (T)1 : (T)0;
}