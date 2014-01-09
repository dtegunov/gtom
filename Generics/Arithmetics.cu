#include "../Prerequisites.cuh"
#include "../Functions.cuh"


////////////////////////////
//CUDA kernel declarations//
////////////////////////////

template <class T> __global__ void MultiplyByVectorKernel(T* d_input, T* multiplicators, T* d_output, size_t elements, int batch);
template <class T> __global__ void MultiplyByScalarKernel(T* d_input, T* d_output, size_t elements, T multiplicator);
template <class T> __global__ void MultiplyByScalarKernel(T* d_input, T* multiplicators, T* d_output, size_t elements);

__global__ void ComplexMultiplyByVectorKernel(tcomplex* d_input, tfloat* multiplicators, tcomplex* d_output, size_t elements, int batch);
__global__ void ComplexMultiplyByScalarKernel(tcomplex* d_input, tcomplex* d_output, size_t elements, tfloat multiplicator);
__global__ void ComplexMultiplyByScalarKernel(tcomplex* d_input, tfloat* multiplicators, tcomplex* d_output, size_t elements);

__global__ void ComplexMultiplyByVectorKernel(tcomplex* d_input, tcomplex* multiplicators, tcomplex* d_output, size_t elements, int batch);
__global__ void ComplexMultiplyByScalarKernel(tcomplex* d_input, tcomplex* d_output, size_t elements, tcomplex multiplicator);
__global__ void ComplexMultiplyByScalarKernel(tcomplex* d_input, tcomplex* multiplicators, tcomplex* d_output, size_t elements);

__global__ void ComplexMultiplyByConjVectorKernel(tcomplex* d_input, tcomplex* multiplicators, tcomplex* d_output, size_t elements, int batch);
__global__ void ComplexMultiplyByConjScalarKernel(tcomplex* d_input, tcomplex* d_output, size_t elements, tcomplex multiplicator);
__global__ void ComplexMultiplyByConjScalarKernel(tcomplex* d_input, tcomplex* multiplicators, tcomplex* d_output, size_t elements);

template <class T> __global__ void AddVectorKernel(T* d_input, T* d_summands, T* d_output, size_t elements, int batch);
template <class T> __global__ void AddScalarKernel(T* d_input, T* d_output, size_t elements, T summand);
template <class T> __global__ void AddScalarKernel(T* d_input, T* d_summands, T* d_output, size_t elements);

template <class T> __global__ void SubtractVectorKernel(T* d_input, T* d_subtrahends, T* d_output, size_t elements, int batch);
template <class T> __global__ void SubtractScalarKernel(T* d_input, T* d_output, size_t elements, T subtrahend);
template <class T> __global__ void SubtractScalarKernel(T* d_input, T* d_subtrahends, T* d_output, size_t elements);

template <class T> __global__ void SquareKernel(T* d_input, T* d_output, size_t elements);
template <class T> __global__ void SqrtKernel(T* d_input, T* d_output, size_t elements);
template <class T> __global__ void PowKernel(T* d_input, T* d_output, size_t elements, T exponent);
template <class T> __global__ void AbsKernel(T* d_input, T* d_output, size_t elements);

template <class T> __global__ void MaxOpKernel(T* d_input1, T* d_input2, T* d_output, size_t elements);
template <class T> __global__ void MinOpKernel(T* d_input1, T* d_input2, T* d_output, size_t elements);


//////////////////
//Multiplication//
//////////////////

template <class T> void d_MultiplyByVector(T* d_input, T* d_multiplicators, T* d_output, size_t elements, int batch)
{
	size_t TpB = min(256, elements);
	size_t totalblocks = min((elements + TpB - 1) / TpB, 32768);
	dim3 grid = dim3((uint)totalblocks);
	MultiplyByVectorKernel<T> <<<grid, (uint)TpB>>> (d_input, d_multiplicators, d_output, elements, batch);
	cudaStreamQuery(0);
}
template void d_MultiplyByVector<tfloat>(tfloat* d_input, tfloat* d_multiplicators, tfloat* d_output, size_t elements, int batch);
template void d_MultiplyByVector<int>(int* d_input, int* d_multiplicators, int* d_output, size_t elements, int batch);

template <class T> void d_MultiplyByScalar(T* d_input, T* d_output, size_t elements, T multiplicator)
{
	size_t TpB = min(256, elements);
	size_t totalblocks = min((elements + TpB - 1) / TpB, 32768);
	dim3 grid = dim3((uint)totalblocks);
	MultiplyByScalarKernel<T> <<<grid, (uint)TpB>>> (d_input, d_output, elements, multiplicator);
	cudaStreamQuery(0);
}
template void d_MultiplyByScalar<tfloat>(tfloat* d_input, tfloat* d_output, size_t elements, tfloat multiplicator);
template void d_MultiplyByScalar<double>(double* d_input, double* d_output, size_t elements, double multiplicator);
template void d_MultiplyByScalar<int>(int* d_input, int* d_output, size_t elements, int multiplicator);

template <class T> void d_MultiplyByScalar(T* d_input, T* d_multiplicators, T* d_output, size_t elements, int batch)
{
	size_t TpB = min(256, elements);
	size_t totalblocks = min((elements + TpB - 1) / TpB, 32768);
	dim3 grid = dim3((uint)totalblocks, batch);
	MultiplyByScalarKernel<T> <<<grid, (uint)TpB>>> (d_input, d_multiplicators, d_output, elements);
	cudaStreamQuery(0);
}
template void d_MultiplyByScalar<tfloat>(tfloat* d_input, tfloat* d_multiplicators, tfloat* d_output, size_t elements, int batch);
template void d_MultiplyByScalar<int>(int* d_input, int* d_multiplicators, int* d_output, size_t elements, int batch);

template <class T> __global__ void MultiplyByVectorKernel(T* d_input, T* d_multiplicators, T* d_output, size_t elements, int batch)
{
	T val;
	for(size_t id = blockIdx.x * blockDim.x + threadIdx.x; 
		id < elements; 
		id += blockDim.x * gridDim.x)
	{
		val = d_multiplicators[id];
		for(size_t n = 0; n < batch; n++)
			d_output[id + elements * n] = d_input[id + elements * n] * val;
	}
}

template <class T> __global__ void MultiplyByScalarKernel(T* d_input, T* d_output, size_t elements, T multiplicator)
{
	for(size_t id = blockIdx.x * blockDim.x + threadIdx.x; 
		id < elements; 
		id += blockDim.x * gridDim.x)
		d_output[id] = d_input[id] * multiplicator;
}

template <class T> __global__ void MultiplyByScalarKernel(T* d_input, T* d_multiplicators, T* d_output, size_t elements)
{
	__shared__ T scalar;
	if(threadIdx.x == 0)
		scalar = d_multiplicators[blockIdx.y];
	__syncthreads();

	size_t offset = elements * blockIdx.y;
	for(size_t id = blockIdx.x * blockDim.x + threadIdx.x; 
		id < elements; 
		id += blockDim.x * gridDim.x)
		d_output[id + offset] = d_input[id + offset] * scalar;
}


//////////////////////////
//Complex Multiplication//
//////////////////////////

void d_ComplexMultiplyByVector(tcomplex* d_input, tfloat* d_multiplicators, tcomplex* d_output, size_t elements, int batch)
{
	size_t TpB = min(256, elements);
	size_t totalblocks = min((elements + TpB - 1) / TpB, 32768);
	dim3 grid = dim3((uint)totalblocks);
	ComplexMultiplyByVectorKernel <<<grid, (uint)TpB>>> (d_input, d_multiplicators, d_output, elements, batch);
	cudaStreamQuery(0);
}

void d_ComplexMultiplyByVector(tcomplex* d_input, tcomplex* d_multiplicators, tcomplex* d_output, size_t elements, int batch)
{
	size_t TpB = min(256, elements);
	size_t totalblocks = min((elements + TpB - 1) / TpB, 32768);
	dim3 grid = dim3((uint)totalblocks);
	ComplexMultiplyByVectorKernel <<<grid, (uint)TpB>>> (d_input, d_multiplicators, d_output, elements, batch);
	cudaStreamQuery(0);
}

void d_ComplexMultiplyByConjVector(tcomplex* d_input, tcomplex* d_multiplicators, tcomplex* d_output, size_t elements, int batch)
{
	size_t TpB = min(256, elements);
	size_t totalblocks = min((elements + TpB - 1) / TpB, 32768);
	dim3 grid = dim3((uint)totalblocks);
	ComplexMultiplyByConjVectorKernel <<<grid, (uint)TpB>>> (d_input, d_multiplicators, d_output, elements, batch);
	cudaStreamQuery(0);
}

void d_ComplexMultiplyByScalar(tcomplex* d_input, tcomplex* d_output, size_t elements, tfloat multiplicator)
{
	size_t TpB = min(256, elements);
	size_t totalblocks = min((elements + TpB - 1) / TpB, 32768);
	dim3 grid = dim3((uint)totalblocks);
	ComplexMultiplyByScalarKernel <<<grid, (uint)TpB>>> (d_input, d_output, elements, multiplicator);
	cudaStreamQuery(0);
}

void d_ComplexMultiplyByScalar(tcomplex* d_input, tcomplex* d_output, size_t elements, tcomplex multiplicator)
{
	size_t TpB = min(256, elements);
	size_t totalblocks = min((elements + TpB - 1) / TpB, 32768);
	dim3 grid = dim3((uint)totalblocks);
	ComplexMultiplyByScalarKernel <<<grid, (uint)TpB>>> (d_input, d_output, elements, multiplicator);
	cudaStreamQuery(0);
}

void d_ComplexMultiplyByConjScalar(tcomplex* d_input, tcomplex* d_output, size_t elements, tcomplex multiplicator)
{
	size_t TpB = min(256, elements);
	size_t totalblocks = min((elements + TpB - 1) / TpB, 32768);
	dim3 grid = dim3((uint)totalblocks);
	ComplexMultiplyByConjScalarKernel <<<grid, (uint)TpB>>> (d_input, d_output, elements, multiplicator);
	cudaStreamQuery(0);
}

void d_ComplexMultiplyByScalar(tcomplex* d_input, tfloat* d_multiplicators, tcomplex* d_output, size_t elements, int batch)
{
	size_t TpB = min(256, elements);
	size_t totalblocks = min((elements + TpB - 1) / TpB, 32768);
	dim3 grid = dim3((uint)totalblocks, batch);
	ComplexMultiplyByScalarKernel <<<grid, (uint)TpB>>> (d_input, d_multiplicators, d_output, elements);
	cudaStreamQuery(0);
}

void d_ComplexMultiplyByScalar(tcomplex* d_input, tcomplex* d_multiplicators, tcomplex* d_output, size_t elements, int batch)
{
	size_t TpB = min(256, elements);
	size_t totalblocks = min((elements + TpB - 1) / TpB, 32768);
	dim3 grid = dim3((uint)totalblocks, batch);
	ComplexMultiplyByScalarKernel <<<grid, (uint)TpB>>> (d_input, d_multiplicators, d_output, elements);
	cudaStreamQuery(0);
}

void d_ComplexMultiplyByConjScalar(tcomplex* d_input, tcomplex* d_multiplicators, tcomplex* d_output, size_t elements, int batch)
{
	size_t TpB = min(256, elements);
	size_t totalblocks = min((elements + TpB - 1) / TpB, 32768);
	dim3 grid = dim3((uint)totalblocks, batch);
	ComplexMultiplyByConjScalarKernel <<<grid, (uint)TpB>>> (d_input, d_multiplicators, d_output, elements);
	cudaStreamQuery(0);
}

__global__ void ComplexMultiplyByVectorKernel(tcomplex* d_input, tfloat* d_multiplicators, tcomplex* d_output, size_t elements, int batch)
{
	tfloat val;
	for(size_t id = blockIdx.x * blockDim.x + threadIdx.x; 
		id < elements; 
		id += blockDim.x * gridDim.x)
	{
		val = d_multiplicators[id];
		for(size_t n = 0; n < batch; n++)
		{
			d_output[id + elements * n].x = d_input[id + elements * n].x * val;
			d_output[id + elements * n].y = d_input[id + elements * n].y * val;
		}
	}
}

__global__ void ComplexMultiplyByVectorKernel(tcomplex* d_input, tcomplex* d_multiplicators, tcomplex* d_output, size_t elements, int batch)
{
	tcomplex val;
	for(size_t id = blockIdx.x * blockDim.x + threadIdx.x; 
		id < elements; 
		id += blockDim.x * gridDim.x)
	{
		val = d_multiplicators[id];
		for(size_t n = 0; n < batch; n++)
		{
			d_output[id + elements * n] = cmul(d_input[id + elements * n], val);
		}
	}
}

__global__ void ComplexMultiplyByConjVectorKernel(tcomplex* d_input, tcomplex* d_multiplicators, tcomplex* d_output, size_t elements, int batch)
{
	tcomplex val;
	for(size_t id = blockIdx.x * blockDim.x + threadIdx.x; 
		id < elements; 
		id += blockDim.x * gridDim.x)
	{
		val = d_multiplicators[id];
		for(size_t n = 0; n < batch; n++)
		{
			d_output[id + elements * n] = cmul(d_input[id + elements * n], cconj(val));
		}
	}
}

__global__ void ComplexMultiplyByScalarKernel(tcomplex* d_input, tcomplex* d_output, size_t elements, tfloat multiplicator)
{
	for(size_t id = blockIdx.x * blockDim.x + threadIdx.x; 
		id < elements; 
		id += blockDim.x * gridDim.x)
	{
		d_output[id].x = d_input[id].x * multiplicator;
		d_output[id].y = d_input[id].y * multiplicator;
	}
}

__global__ void ComplexMultiplyByScalarKernel(tcomplex* d_input, tcomplex* d_output, size_t elements, tcomplex multiplicator)
{
	for(size_t id = blockIdx.x * blockDim.x + threadIdx.x; 
		id < elements; 
		id += blockDim.x * gridDim.x)
		d_output[id] = cmul(d_input[id], multiplicator);
}

__global__ void ComplexMultiplyByConjScalarKernel(tcomplex* d_input, tcomplex* d_output, size_t elements, tcomplex multiplicator)
{
	for(size_t id = blockIdx.x * blockDim.x + threadIdx.x; 
		id < elements; 
		id += blockDim.x * gridDim.x)
		d_output[id] = cmul(d_input[id], cconj(multiplicator));
}

__global__ void ComplexMultiplyByScalarKernel(tcomplex* d_input, tfloat* d_multiplicators, tcomplex* d_output, size_t elements)
{
	__shared__ tfloat scalar;
	if(threadIdx.x == 0)
		scalar = d_multiplicators[blockIdx.y];
	__syncthreads();

	size_t offset = elements * blockIdx.y;
	for(size_t id = blockIdx.x * blockDim.x + threadIdx.x; 
		id < elements; 
		id += blockDim.x * gridDim.x)
	{
		d_output[id + offset].x = d_input[id + offset].x * scalar;
		d_output[id + offset].y = d_input[id + offset].y * scalar;
	}
}

__global__ void ComplexMultiplyByScalarKernel(tcomplex* d_input, tcomplex* d_multiplicators, tcomplex* d_output, size_t elements)
{
	__shared__ tcomplex scalar;
	if(threadIdx.x == 0)
		scalar = d_multiplicators[blockIdx.y];
	__syncthreads();

	size_t offset = elements * blockIdx.y;
	for(size_t id = blockIdx.x * blockDim.x + threadIdx.x; 
		id < elements; 
		id += blockDim.x * gridDim.x)
		d_output[id + offset] = cmul(d_input[id + offset], scalar);
}

__global__ void ComplexMultiplyByConjScalarKernel(tcomplex* d_input, tcomplex* d_multiplicators, tcomplex* d_output, size_t elements)
{
	__shared__ tcomplex scalar;
	if(threadIdx.x == 0)
		scalar = d_multiplicators[blockIdx.y];
	__syncthreads();

	size_t offset = elements * blockIdx.y;
	for(size_t id = blockIdx.x * blockDim.x + threadIdx.x; 
		id < elements; 
		id += blockDim.x * gridDim.x)
		d_output[id + offset] = cmul(d_input[id + offset], cconj(scalar));
}


////////////
//Addition//
////////////

template <class T> void d_AddVector(T* d_input, T* d_summands, T* d_output, size_t elements, int batch)
{
	size_t TpB = min(256, elements);
	size_t totalblocks = min((elements + TpB - 1) / TpB, 32768);
	dim3 grid = dim3((uint)totalblocks);
	AddVectorKernel<T> <<<grid, (uint)TpB>>> (d_input, d_summands, d_output, elements, batch);
}
template void d_AddVector<tfloat>(tfloat* d_input, tfloat* d_summands, tfloat* d_output, size_t elements, int batch);
template void d_AddVector<int>(int* d_input, int* d_summands, int* d_output, size_t elements, int batch);

template <class T> void d_AddScalar(T* d_input, T* d_output, size_t elements, T summand)
{
	size_t TpB = min(256, elements);
	size_t totalblocks = min((elements + TpB - 1) / TpB, 32768);
	dim3 grid = dim3((uint)totalblocks);
	AddScalarKernel<T> <<<grid, (uint)TpB>>> (d_input, d_output, elements, summand);
}
template void d_AddScalar<tfloat>(tfloat* d_input, tfloat* d_output, size_t elements, tfloat summand);
template void d_AddScalar<int>(int* d_input, int* d_output, size_t elements, int summand);

template <class T> void d_AddScalar(T* d_input, T* d_summands, T* d_output, size_t elements, int batch)
{
	size_t TpB = min(256, elements);
	size_t totalblocks = min((elements + TpB - 1) / TpB, 32768);
	dim3 grid = dim3((uint)totalblocks, batch);
	AddScalarKernel<T> <<<grid, (uint)TpB>>> (d_input, d_summands, d_output, elements);
}
template void d_AddScalar<tfloat>(tfloat* d_input, tfloat* d_summands, tfloat* d_output, size_t elements, int batch);
template void d_AddScalar<int>(int* d_input, int* d_summands, int* d_output, size_t elements, int batch);

template <class T> __global__ void AddVectorKernel(T* d_input, T* d_summands, T* d_output, size_t elements, int batch)
{
	T val;
	for(size_t id = blockIdx.x * blockDim.x + threadIdx.x; 
		id < elements; 
		id += blockDim.x * gridDim.x)
	{
		val = d_summands[id];
		for(size_t n = 0; n < batch; n++)
			d_output[id + elements * n] = d_input[id + elements * n] + val;
	}
}

template <class T> __global__ void AddScalarKernel(T* d_input, T* d_output, size_t elements, T summand)
{
	for(size_t id = blockIdx.x * blockDim.x + threadIdx.x; 
		id < elements; 
		id += blockDim.x * gridDim.x)
		d_output[id] = d_input[id] + summand;
}

template <class T> __global__ void AddScalarKernel(T* d_input, T* d_summands, T* d_output, size_t elements)
{
	__shared__ T scalar;
	if(threadIdx.x == 0)
		scalar = d_summands[blockIdx.y];
	__syncthreads();

	size_t offset = elements * blockIdx.y;
	for(size_t id = blockIdx.x * blockDim.x + threadIdx.x; 
		id < elements; 
		id += blockDim.x * gridDim.x)
		d_output[id + offset] = d_input[id + offset] + scalar;
}


///////////////
//Subtraction//
///////////////

template <class T> void d_SubtractVector(T* d_input, T* d_subtrahends, T* d_output, size_t elements, int batch)
{
	size_t TpB = min(256, elements);
	size_t totalblocks = min((elements + TpB - 1) / TpB, 32768);
	dim3 grid = dim3((uint)totalblocks);
	SubtractVectorKernel<T> <<<grid, (uint)TpB>>> (d_input, d_subtrahends, d_output, elements, batch);
}
template void d_SubtractVector<tfloat>(tfloat* d_input, tfloat* d_subtrahends, tfloat* d_output, size_t elements, int batch);
template void d_SubtractVector<int>(int* d_input, int* d_subtrahends, int* d_output, size_t elements, int batch);

template <class T> void d_SubtractScalar(T* d_input, T* d_output, size_t elements, T subtrahend)
{
	size_t TpB = min(256, elements);
	size_t totalblocks = min((elements + TpB - 1) / TpB, 32768);
	dim3 grid = dim3((uint)totalblocks);
	SubtractScalarKernel<T> <<<grid, (uint)TpB>>> (d_input, d_output, elements, subtrahend);
}
template void d_SubtractScalar<tfloat>(tfloat* d_input, tfloat* d_output, size_t elements, tfloat subtrahend);
template void d_SubtractScalar<int>(int* d_input, int* d_output, size_t elements, int subtrahend);

template <class T> void d_SubtractScalar(T* d_input, T* d_subtrahends, T* d_output, size_t elements, int batch)
{
	size_t TpB = min(256, elements);
	size_t totalblocks = min((elements + TpB - 1) / TpB, 32768);
	dim3 grid = dim3((uint)totalblocks, batch);
	SubtractScalarKernel<T> <<<grid, (uint)TpB>>> (d_input, d_subtrahends, d_output, elements);
}
template void d_SubtractScalar<tfloat>(tfloat* d_input, tfloat* d_subtrahends, tfloat* d_output, size_t elements, int batch);
template void d_SubtractScalar<int>(int* d_input, int* d_subtrahends, int* d_output, size_t elements, int batch);

template <class T> __global__ void SubtractVectorKernel(T* d_input, T* d_subtrahends, T* d_output, size_t elements, int batch)
{
	T val;
	for(size_t id = blockIdx.x * blockDim.x + threadIdx.x; 
		id < elements; 
		id += blockDim.x * gridDim.x)
	{
		val = d_subtrahends[id];
		for(size_t n = 0; n < batch; n++)
			d_output[id + elements * n] = d_input[id + elements * n] - val;
	}
}

template <class T> __global__ void SubtractScalarKernel(T* d_input, T* d_output, size_t elements, T subtrahend)
{
	for(size_t id = blockIdx.x * blockDim.x + threadIdx.x; 
		id < elements; 
		id += blockDim.x * gridDim.x)
		d_output[id] = d_input[id] - subtrahend;
}

template <class T> __global__ void SubtractScalarKernel(T* d_input, T* d_subtrahends, T* d_output, size_t elements)
{
	__shared__ T scalar;
	if(threadIdx.x == 0)
		scalar = d_subtrahends[blockIdx.y];
	__syncthreads();

	size_t offset = elements * blockIdx.y;
	size_t gridsize = blockDim.x * gridDim.x;
	for(size_t id = blockIdx.x * blockDim.x + threadIdx.x; 
		id < elements; 
		id += gridsize)
		d_output[id + offset] = d_input[id + offset] - scalar;
}


//////////
//Square//
//////////

template <class T> void d_Square(T* d_input, T* d_output, size_t elements, int batch)
{
	size_t TpB = min(256, elements);
	size_t totalblocks = min((elements + TpB - 1) / TpB, 32768);
	dim3 grid = dim3((uint)totalblocks);
	for(int b = 0; b < batch; b++)
		SquareKernel<T> <<<grid, (uint)TpB>>> (d_input + elements * b, d_output + elements * b, elements);
}
template void d_Square<tfloat>(tfloat* d_input, tfloat* d_output, size_t elements, int batch);
template void d_Square<int>(int* d_input, int* d_output, size_t elements, int batch);

template <class T> __global__ void SquareKernel(T* d_input, T* d_output, size_t elements)
{
	T val;
	int gridsize = blockDim.x * gridDim.x;
	for(int id = blockIdx.x * blockDim.x + threadIdx.x; 
		id < elements; 
		id += gridsize)
	{
		val = d_input[id];
		d_output[id] = val * val;
	}
}


///////////////
//Square root//
///////////////

template <class T> void d_Sqrt(T* d_input, T* d_output, size_t elements)
{
	size_t TpB = min(256, elements);
	size_t totalblocks = min((elements + TpB - 1) / TpB, 8192);
	dim3 grid = dim3((uint)totalblocks);
	SqrtKernel<T> <<<grid, (uint)TpB>>> (d_input, d_output, elements);
}
template void d_Sqrt<tfloat>(tfloat* d_input, tfloat* d_output, size_t elements);
//template void d_Sqrt<int>(int* d_input, int* d_output, size_t elements);

template <class T> __global__ void SqrtKernel(T* d_input, T* d_output, size_t elements)
{
	for(size_t id = blockIdx.x * blockDim.x + threadIdx.x; 
		id < elements; 
		id += blockDim.x * gridDim.x)
		d_output[id] = sqrt(d_input[id]);
}


/////////
//Power//
/////////

template <class T> void d_Pow(T* d_input, T* d_output, size_t elements, T exponent)
{
	size_t TpB = min(256, elements);
	size_t totalblocks = min((elements + TpB - 1) / TpB, 8192);
	dim3 grid = dim3((uint)totalblocks);
	PowKernel<T> <<<grid, (uint)TpB>>> (d_input, d_output, elements, exponent);
}
template void d_Pow<tfloat>(tfloat* d_input, tfloat* d_output, size_t elements, tfloat exponent);

template <class T> __global__ void PowKernel(T* d_input, T* d_output, size_t elements, T exponent)
{
	for(size_t id = blockIdx.x * blockDim.x + threadIdx.x; 
		id < elements; 
		id += blockDim.x * gridDim.x)
		d_output[id] = pow(d_input[id], exponent);
}


///////
//Abs//
///////

template <class T> void d_Abs(T* d_input, T* d_output, size_t elements)
{
	size_t TpB = min(256, elements);
	size_t totalblocks = min((elements + TpB - 1) / TpB, 8192);
	dim3 grid = dim3((uint)totalblocks);
	AbsKernel<T> <<<grid, (uint)TpB>>> (d_input, d_output, elements);
}
template void d_Abs<tfloat>(tfloat* d_input, tfloat* d_output, size_t elements);

template <class T> __global__ void AbsKernel(T* d_input, T* d_output, size_t elements)
{
	for(size_t id = blockIdx.x * blockDim.x + threadIdx.x; 
		id < elements; 
		id += blockDim.x * gridDim.x)
		d_output[id] = abs(d_input[id]);
}


///////////////
//Min/Max ops//
///////////////

template <class T> void d_MaxOp(T* d_input1, T* d_input2, T* d_output, size_t elements)
{
	size_t TpB = min(256, elements);
	size_t totalblocks = min((elements + TpB - 1) / TpB, 8192);
	dim3 grid = dim3((uint)totalblocks);
	MaxOpKernel<T> <<<grid, (uint)TpB>>> (d_input1, d_input2, d_output, elements);
}
template void d_MaxOp<int>(int* d_input1, int* d_input2, int* d_output, size_t elements);
template void d_MaxOp<float>(float* d_input1, float* d_input2, float* d_output, size_t elements);
template void d_MaxOp<double>(double* d_input1, double* d_input2, double* d_output, size_t elements);

template <class T> void d_MinOp(T* d_input1, T* d_input2, T* d_output, size_t elements)
{
	size_t TpB = min(256, elements);
	size_t totalblocks = min((elements + TpB - 1) / TpB, 8192);
	dim3 grid = dim3((uint)totalblocks);
	MinOpKernel<T> <<<grid, (uint)TpB>>> (d_input1, d_input2, d_output, elements);
}
template void d_MinOp<int>(int* d_input1, int* d_input2, int* d_output, size_t elements);
template void d_MinOp<float>(float* d_input1, float* d_input2, float* d_output, size_t elements);
template void d_MinOp<double>(double* d_input1, double* d_input2, double* d_output, size_t elements);

template <class T> __global__ void MaxOpKernel(T* d_input1, T* d_input2, T* d_output, size_t elements)
{
	for(size_t id = blockIdx.x * blockDim.x + threadIdx.x; 
		id < elements; 
		id += blockDim.x * gridDim.x)
		d_output[id] = max(d_input1[id], d_input2[id]);
}

template <class T> __global__ void MinOpKernel(T* d_input1, T* d_input2, T* d_output, size_t elements)
{
	for(size_t id = blockIdx.x * blockDim.x + threadIdx.x; 
		id < elements; 
		id += blockDim.x * gridDim.x)
		d_output[id] = min(d_input1[id], d_input2[id]);
}


////////
//Misc//
////////

size_t NextPow2(size_t x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

bool IsPow2(size_t x) 
{
	return x && !(x & (x - 1));
}