#include "..\Prerequisites.cuh"
#include "..\Functions.cuh"


////////////////////////////
//CUDA kernel declarations//
////////////////////////////

template <class T> __global__ void ValueFillKernel(T* d_output, size_t elements, T value);
template <class T, int fieldcount> __global__ void JoinInterleavedKernel(T** d_fields, T* d_output, size_t elements);


///////////////
//Host memory//
///////////////

void* MallocFromDeviceArray(void* d_array, size_t size)
{
	void* h_array = malloc(size);
	cudaMemcpy(h_array, d_array, size, cudaMemcpyDeviceToHost);

	return h_array;
}

tfloat* MallocZeroFilledFloat(size_t elements)
{
	return MallocValueFilled<tfloat>(elements, (tfloat)0.0);
}

template <class T> T* MallocValueFilled(size_t elements, T value)
{
	T* h_array = (T*)malloc(elements * sizeof(T));

	intptr_t s_elements = (intptr_t)elements;
	#pragma omp for schedule(dynamic, 1024)
	for(intptr_t i = 0; i < s_elements; i++)
		h_array[i] = value;

	return h_array;
}
template tfloat* MallocValueFilled<tfloat>(size_t elements, tfloat value);
template double* MallocValueFilled<double>(size_t elements, double value);
template tcomplex* MallocValueFilled<tcomplex>(size_t elements, tcomplex value);
template char* MallocValueFilled<char>(size_t elements, char value);
template bool* MallocValueFilled<bool>(size_t elements, bool value);
template int* MallocValueFilled<int>(size_t elements, int value);


/////////////////
//Device memory//
/////////////////

void* CudaMallocFromHostArray(void* h_array, size_t size)
{
	void* d_array;
	cudaMalloc((void**)&d_array, size);
	cudaMemcpy(d_array, h_array, size, cudaMemcpyHostToDevice);

	return d_array;
}

void* CudaMallocFromHostArray(void* h_array, size_t devicesize, size_t hostsize)
{
	void* d_array;
	cudaMalloc((void**)&d_array, devicesize);
	cudaMemcpy(d_array, h_array, hostsize, cudaMemcpyHostToDevice);

	return d_array;
}

tfloat* CudaMallocZeroFilledFloat(size_t elements)
{
	return CudaMallocValueFilled<tfloat>(elements, (tfloat)0.0);
}

template <class T> T* CudaMallocValueFilled(size_t elements, T value)
{
	T* d_array;
	cudaMalloc((void**)&d_array, elements * sizeof(T));

	size_t TpB = 256;
	size_t totalblocks = min((elements + TpB - 1) / TpB, 8192);
	dim3 grid = dim3((uint)totalblocks);
	ValueFillKernel<T> <<<grid, (uint)TpB>>> (d_array, elements, value);

	return d_array;
}
template tfloat* CudaMallocValueFilled<tfloat>(size_t elements, tfloat value);
template tcomplex* CudaMallocValueFilled<tcomplex>(size_t elements, tcomplex value);
template char* CudaMallocValueFilled<char>(size_t elements, char value);
template bool* CudaMallocValueFilled<bool>(size_t elements, bool value);
template int* CudaMallocValueFilled<int>(size_t elements, int value);

template <class T, int fieldcount> T* d_JoinInterleaved(T** d_fields, size_t elements)
{
	T* d_output;
	cudaMalloc((void**)&d_output, elements * fieldcount * sizeof(T));

	size_t TpB = 256;
	size_t totalblocks = min((elements + TpB - 1) / TpB, 8192);
	dim3 grid = dim3((uint)totalblocks);
	JoinInterleavedKernel<T, fieldcount> <<<grid, (uint)TpB>>> (d_fields, d_output, elements);

	cudaDeviceSynchronize();

	return d_output;
}
template tfloat* d_JoinInterleaved<tfloat, 2>(tfloat** d_fields, size_t elements);
template tfloat* d_JoinInterleaved<tfloat, 3>(tfloat** d_fields, size_t elements);
template tfloat* d_JoinInterleaved<tfloat, 4>(tfloat** d_fields, size_t elements);
template tfloat* d_JoinInterleaved<tfloat, 5>(tfloat** d_fields, size_t elements);
template tfloat* d_JoinInterleaved<tfloat, 6>(tfloat** d_fields, size_t elements);
template int* d_JoinInterleaved<int, 2>(int** d_fields, size_t elements);
template int* d_JoinInterleaved<int, 3>(int** d_fields, size_t elements);
template int* d_JoinInterleaved<int, 4>(int** d_fields, size_t elements);
template int* d_JoinInterleaved<int, 5>(int** d_fields, size_t elements);
template int* d_JoinInterleaved<int, 6>(int** d_fields, size_t elements);

template <class T, int fieldcount> void d_JoinInterleaved(T** d_fields, T* d_output, size_t elements)
{
	size_t TpB = 256;
	size_t totalblocks = min((elements + TpB - 1) / TpB, 8192);
	dim3 grid = dim3((uint)totalblocks);
	JoinInterleavedKernel<T, fieldcount> <<<grid, (uint)TpB>>> (d_fields, d_output, elements);

	cudaDeviceSynchronize();
}
template void d_JoinInterleaved<tfloat, 2>(tfloat** d_fields, tfloat* d_output, size_t elements);
template void d_JoinInterleaved<tfloat, 3>(tfloat** d_fields, tfloat* d_output, size_t elements);
template void d_JoinInterleaved<tfloat, 4>(tfloat** d_fields, tfloat* d_output, size_t elements);
template void d_JoinInterleaved<tfloat, 5>(tfloat** d_fields, tfloat* d_output, size_t elements);
template void d_JoinInterleaved<tfloat, 6>(tfloat** d_fields, tfloat* d_output, size_t elements);
template void d_JoinInterleaved<int, 2>(int** d_fields, int* d_output, size_t elements);
template void d_JoinInterleaved<int, 3>(int** d_fields, int* d_output, size_t elements);
template void d_JoinInterleaved<int, 4>(int** d_fields, int* d_output, size_t elements);
template void d_JoinInterleaved<int, 5>(int** d_fields, int* d_output, size_t elements);
template void d_JoinInterleaved<int, 6>(int** d_fields, int* d_output, size_t elements);

////////////////
//CUDA kernels//
////////////////

template <class T> __global__ void ValueFillKernel(T* d_output, size_t elements, T value)
{
	for(size_t id = blockIdx.x * blockDim.x + threadIdx.x; 
		id < elements; 
		id += blockDim.x * gridDim.x)
		d_output[id] = value;
}

template <class T, int fieldcount> __global__ void JoinInterleavedKernel(T** d_fields, T* d_output, size_t elements)
{
	size_t startid = blockIdx.x * blockDim.x + threadIdx.x;
	int gridsize = blockDim.x * gridDim.x;

	//Roll out the first 5 iterations and put the rest into a for-loop since more than 5 fields are unlikely.
	//Maybe the compiler is smart enough to roll out the rest itself.
	if(fieldcount > 0)
	{
		T* d_field = d_fields[0];
		for(size_t id = startid; 
			id < elements; 
			id += gridsize)
			d_output[id * fieldcount] = d_field[id];
	}

	if(fieldcount > 1)
	{
		T* d_field = d_fields[1];
		for(size_t id = startid; 
			id < elements; 
			id += gridsize)
			d_output[id * fieldcount + 1] = d_field[id];
	}

	if(fieldcount > 2)
	{
		T* d_field = d_fields[2];
		for(size_t id = startid; 
			id < elements; 
			id += gridsize)
			d_output[id * fieldcount + 2] = d_field[id];
	}

	if(fieldcount > 3)
	{
		T* d_field = d_fields[3];
		for(size_t id = startid; 
			id < elements; 
			id += gridsize)
			d_output[id * fieldcount + 3] = d_field[id];
	}

	if(fieldcount > 4)
	{
		T* d_field = d_fields[4];
		for(size_t id = startid; 
			id < elements; 
			id += gridsize)
			d_output[id * fieldcount + 4] = d_field[id];
	}

	if(fieldcount > 5)
	{
		for(int f = 5; f < fieldcount; f++)
		{
			T* d_field = d_fields[f];
			for(size_t id = startid; 
				id < elements; 
				id += gridsize)
				d_output[id * fieldcount + f] = d_field[id];
		}
	}
}