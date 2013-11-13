#include "../Prerequisites.cuh"
#include "../Functions.cuh"


////////////////////////////
//CUDA kernel declarations//
////////////////////////////

template <class T> __global__ void RemapKernel(T* d_input, intptr_t* d_map, T* d_output, size_t elementsmapped, size_t elementsoriginal, T defvalue, int batch);


//////////////////
//Data remapping//
//////////////////

template <class T> void d_Remap(T* d_input, intptr_t* d_map, T* d_output, size_t elementsmapped, size_t elementsoriginal, T defvalue, int batch)
{
	size_t TpB = 192;
	size_t totalblocks = min((elementsmapped + TpB - 1) / TpB, 32768);
	dim3 grid = dim3((uint)totalblocks);
	RemapKernel<T> <<<grid, (uint)TpB>>> (d_input, d_map, d_output, elementsmapped, elementsoriginal, defvalue, batch);
}
template void d_Remap<tfloat>(tfloat* d_input, intptr_t* d_map, tfloat* d_output, size_t elementsmapped, size_t elementsoriginal, tfloat defvalue, int batch);
template void d_Remap<int>(int* d_input, intptr_t* d_map, int* d_output, size_t elementsmapped, size_t elementsoriginal, int defvalue, int batch);

template <class T> void Remap(T* h_input, intptr_t* h_map, T* h_output, size_t elementsmapped, size_t elementsoriginal, T defvalue, int batch)
{
	T* d_input = (T*)CudaMallocFromHostArray(h_input, elementsoriginal * batch * sizeof(T));
	intptr_t* d_map = (intptr_t*)CudaMallocFromHostArray(h_map, elementsmapped * sizeof(intptr_t));
	T* d_output;
	cudaMalloc((void**)&d_output, elementsmapped * batch * sizeof(T));

	d_Remap(d_input, d_map, d_output, elementsmapped, elementsoriginal, defvalue, batch);

	cudaMemcpy(h_output, d_output, elementsmapped * batch * sizeof(T), cudaMemcpyDeviceToHost);

	cudaFree(d_input);
	cudaFree(d_map);
	cudaFree(d_output);
}
template void Remap<tfloat>(tfloat* d_input, intptr_t* d_map, tfloat* d_output, size_t elementsmapped, size_t elementsoriginal, tfloat defvalue, int batch);
template void Remap<int>(int* d_input, intptr_t* d_map, int* d_output, size_t elementsmapped, size_t elementsoriginal, int defvalue, int batch);

template <class T> __global__ void RemapKernel(T* d_input, intptr_t* d_map, T* d_output, size_t elementsmapped, size_t elementsoriginal, T defvalue, int batch)
{
	intptr_t address;
	for(size_t id = blockIdx.x * blockDim.x + threadIdx.x; 
		id < elementsmapped; 
		id += blockDim.x * gridDim.x)
	{
		address = d_map[id];
		if(address < 0)
			for(size_t b = 0; b < batch; b++)
				d_output[id + elementsmapped * b] = defvalue;
		else
			for(size_t b = 0; b < batch; b++)
				d_output[id + elementsmapped * b] = d_input[address + elementsoriginal * b];
	}
}


///////////////////////////////////
//Sparse mask to dense conversion//
///////////////////////////////////

template <class T> void MaskSparseToDense(T* h_input, intptr_t** h_mapforward, intptr_t* h_mapbackward, size_t &elementsmapped, size_t elementsoriginal)
{
	intptr_t lastaddress = 0;
	intptr_t* h_tempforward = (intptr_t*)malloc(elementsoriginal * sizeof(intptr_t));

	if(h_mapbackward != NULL)
		for(size_t i = 0; i < elementsoriginal; i++)
			if(h_input[i] > 0)
			{
				h_tempforward[lastaddress] = i;
				h_mapbackward[i] = lastaddress;
				lastaddress++;
			}
			else
				h_mapbackward[i] = -1;
	else
		for(size_t i = 0; i < elementsoriginal; i++)
		{
			if(h_input[i] > 0)
			{
				h_tempforward[lastaddress] = i;
				lastaddress++;
			}
		}

	if(lastaddress == 0)
	{
		*h_mapforward = NULL;
		elementsmapped = 0;
	}
	else
	{
		*h_mapforward = (intptr_t*)malloc(lastaddress * sizeof(intptr_t));
		memcpy(*h_mapforward, h_tempforward, lastaddress * sizeof(intptr_t));
		elementsmapped = lastaddress;
	}

	free(h_tempforward);
}
template void MaskSparseToDense<float>(float* h_input, intptr_t** h_mapforward, intptr_t* h_mapbackward, size_t &elementsmapped, size_t elementsoriginal);
template void MaskSparseToDense<double>(double* h_input, intptr_t** h_mapforward, intptr_t* h_mapbackward, size_t &elementsmapped, size_t elementsoriginal);
template void MaskSparseToDense<int>(int* h_input, intptr_t** h_mapforward, intptr_t* h_mapbackward, size_t &elementsmapped, size_t elementsoriginal);
template void MaskSparseToDense<bool>(bool* h_input, intptr_t** h_mapforward, intptr_t* h_mapbackward, size_t &elementsmapped, size_t elementsoriginal);
template void MaskSparseToDense<char>(char* h_input, intptr_t** h_mapforward, intptr_t* h_mapbackward, size_t &elementsmapped, size_t elementsoriginal);


template <class T> void d_MaskSparseToDense(T* d_input, intptr_t** d_mapforward, intptr_t* d_mapbackward, size_t &elementsmapped, size_t elementsoriginal)
{
	T* h_input = (T*)MallocFromDeviceArray(d_input, elementsoriginal * sizeof(T));
	intptr_t* h_mapforward = NULL;
	intptr_t* h_mapbackward = d_mapbackward == NULL ? NULL : (intptr_t*)malloc(elementsoriginal * sizeof(intptr_t));
	size_t elements = 0;

	MaskSparseToDense(h_input, &h_mapforward, h_mapbackward, elements, elementsoriginal);

	*d_mapforward = h_mapforward == NULL ? NULL : (intptr_t*)CudaMallocFromHostArray(h_mapforward, elements * sizeof(intptr_t));
	if(d_mapbackward != NULL && h_mapbackward != NULL)
		cudaMemcpy(d_mapbackward, h_mapbackward, elementsoriginal * sizeof(intptr_t), cudaMemcpyHostToDevice);

	elementsmapped = elements;

	free(h_input);
	if(h_mapbackward != NULL)
		free(h_mapbackward);
	if(h_mapforward != NULL)
		free(h_mapforward);
}
template void d_MaskSparseToDense<float>(float* h_input, intptr_t** h_mapforward, intptr_t* h_mapbackward, size_t &elementsmapped, size_t elementsoriginal);
template void d_MaskSparseToDense<double>(double* h_input, intptr_t** h_mapforward, intptr_t* h_mapbackward, size_t &elementsmapped, size_t elementsoriginal);
template void d_MaskSparseToDense<int>(int* h_input, intptr_t** h_mapforward, intptr_t* h_mapbackward, size_t &elementsmapped, size_t elementsoriginal);
template void d_MaskSparseToDense<bool>(bool* h_input, intptr_t** h_mapforward, intptr_t* h_mapbackward, size_t &elementsmapped, size_t elementsoriginal);
template void d_MaskSparseToDense<char>(char* h_input, intptr_t** h_mapforward, intptr_t* h_mapbackward, size_t &elementsmapped, size_t elementsoriginal);