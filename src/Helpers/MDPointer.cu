#include "Prerequisites.cuh"
#include "Helper.cuh"


namespace gtom
{
	template <class T> MDPointer<T>::MDPointer()
	{
		devicecount = 0;
		cudaGetDeviceCount(&devicecount);

		pointers = (T**)malloc(devicecount * sizeof(T*));
		for (int i = 0; i < devicecount; i++)
			pointers[i] = NULL;
	}

	template <class T> MDPointer<T>::~MDPointer()
	{
		this->Free();
		//free(pointers);
	}

	template <class T> void MDPointer<T>::Malloc(size_t size)
	{
#pragma omp parallel for
		for (int i = 0; i < devicecount; i++)
		{
			cudaSetDevice(i);
			cudaMalloc((void**)(pointers + i), size);
		}
	}

	template <class T> void MDPointer<T>::Free()
	{
		for (int i = 0; i < devicecount; i++)
			if (pointers[i] != NULL)
			{
				cudaSetDevice(i);
				cudaFree(pointers[i]);
			}
	}

	template <class T>void  MDPointer<T>::MallocFromHostArray(T* h_src, size_t devicesize, size_t hostsize)
	{
#pragma omp parallel for
		for (int i = 0; i < devicecount; i++)
		{
			cudaSetDevice(i);
			pointers[i] = (T*)CudaMallocFromHostArray(h_src, devicesize, hostsize);
		}
	}

	template <class T>void  MDPointer<T>::MallocFromHostArray(T* h_src, size_t size)
	{
		this->MallocFromHostArray(h_src, size, size);
	}

	template <class T> void MDPointer<T>::Memcpy(T* h_src, size_t deviceoffset, size_t hostsize)
	{
#pragma omp parallel for
		for (int i = 0; i < devicecount; i++)
		{
			cudaSetDevice(i);
			cudaMemcpy(pointers[i] + deviceoffset, h_src, hostsize, cudaMemcpyHostToDevice);
		}
	}

	template <class T> void MDPointer<T>::MallocValueFilled(size_t elements, T value)
	{
#pragma omp parallel for
		for (int i = 0; i < devicecount; i++)
		{
			cudaSetDevice(i);
			pointers[i] = CudaMallocValueFilled<T>(elements, value);
		}
	}

	template <class T> bool MDPointer<T>::operator==(const MDPointer<T> &other) const
	{
		return this->pointers == other.pointers;
	}
}