#include "..\Prerequisites.cuh"
#include "..\Functions.cuh"


////////////////////////////
//CUDA kernel declarations//
////////////////////////////

template <class T, uint blockSize, bool nIsPow2> __global__ void SumKernel(T* d_input, T* d_output, size_t n);


///////
//Sum//
///////

void GetNumBlocksAndThreads(size_t n, int &blocks, int &threads, int maxblocks)
{
    //get device capability, to avoid block/grid size excceed the upbound
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
	
    size_t maxthreads = 512;
    threads = (int)((n < maxthreads * 2) ? NextPow2((n + 1)/ 2) : maxthreads);
    size_t totalblocks = (n + (threads * 2 - 1)) / (threads * 2);
    totalblocks = min(maxblocks, totalblocks);
	blocks = (int)totalblocks;
}

template <class T> void SumReduce(T *d_input, T *d_output, size_t n, int blocks, int threads)
{
    dim3 dimBlock = dim3(threads);
    dim3 dimGrid = dim3(blocks);

    // when there is only one warp per block, we need to allocate two warps
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

    // choose which of the optimized versions of reduction to launch
    
    if (IsPow2(n))
        switch (threads)
        {
            case 512:
                SumKernel<T, 512, true> <<<dimGrid, dimBlock, smemSize>>> (d_input, d_output, n); break;
            case 256:
                SumKernel<T, 256, true> <<<dimGrid, dimBlock, smemSize>>> (d_input, d_output, n); break;
            case 128:
                SumKernel<T, 128, true> <<<dimGrid, dimBlock, smemSize>>> (d_input, d_output, n); break;
            case 64:
                SumKernel<T,  64, true> <<<dimGrid, dimBlock, smemSize>>> (d_input, d_output, n); break;
            case 32:
                SumKernel<T,  32, true> <<<dimGrid, dimBlock, smemSize>>> (d_input, d_output, n); break;
            case 16:
                SumKernel<T,  16, true> <<<dimGrid, dimBlock, smemSize>>> (d_input, d_output, n); break;
            case  8:
                SumKernel<T,   8, true> <<<dimGrid, dimBlock, smemSize>>> (d_input, d_output, n); break;
            case  4:
                SumKernel<T,   4, true> <<<dimGrid, dimBlock, smemSize>>> (d_input, d_output, n); break;
            case  2:
                SumKernel<T,   2, true> <<<dimGrid, dimBlock, smemSize>>> (d_input, d_output, n); break;
            case  1:
                SumKernel<T,   1, true> <<<dimGrid, dimBlock, smemSize>>> (d_input, d_output, n); break;
        }
    else
        switch (threads)
        {
            case 512:
                SumKernel<T, 512, false> <<<dimGrid, dimBlock, smemSize>>> (d_input, d_output, n); break;
            case 256:
                SumKernel<T, 256, false> <<<dimGrid, dimBlock, smemSize>>> (d_input, d_output, n); break;
            case 128:
                SumKernel<T, 128, false> <<<dimGrid, dimBlock, smemSize>>> (d_input, d_output, n); break;
            case 64:
                SumKernel<T,  64, false> <<<dimGrid, dimBlock, smemSize>>> (d_input, d_output, n); break;
            case 32:
                SumKernel<T,  32, false> <<<dimGrid, dimBlock, smemSize>>> (d_input, d_output, n); break;
            case 16:
                SumKernel<T,  16, false> <<<dimGrid, dimBlock, smemSize>>> (d_input, d_output, n); break;
            case  8:
                SumKernel<T,   8, false> <<<dimGrid, dimBlock, smemSize>>> (d_input, d_output, n); break;
            case  4:
                SumKernel<T,   4, false> <<<dimGrid, dimBlock, smemSize>>> (d_input, d_output, n); break;
            case  2:
                SumKernel<T,   2, false> <<<dimGrid, dimBlock, smemSize>>> (d_input, d_output, n); break;
            case  1:
                SumKernel<T,   1, false> <<<dimGrid, dimBlock, smemSize>>> (d_input, d_output, n); break;
        }

	cudaDeviceSynchronize();
}

template <class T> void d_Sum(T* d_input, T* d_output, size_t n, int batch)
{
	if(n <= 0)
		return;

	int maxblocks = 512;
	int numblocks = 0;
	int numthreads = 0;
	GetNumBlocksAndThreads(n, numblocks, numthreads, maxblocks);

	T* d_intermediate;
	cudaMalloc((void**)&d_intermediate, numblocks * sizeof(T));

	T* h_intermediate = (T*)malloc(numblocks * sizeof(T));

	for(int b = 0; b < batch; b++)
	{
		SumReduce<T>(d_input + (n * (size_t)b), d_intermediate, n, numblocks, numthreads);
		
		cudaMemcpy(h_intermediate, d_intermediate, numblocks * sizeof(T), cudaMemcpyDeviceToHost);

		T result = h_intermediate[0];
		T c = 0, y, t;
		for (int i = 1; i < numblocks; i++)
		{
			y = h_intermediate[i] - c;
			t = result + y;
			c = (t - result) - y;
			result = t;
		}

		cudaMemcpy(d_output + b, &result, sizeof(T), cudaMemcpyHostToDevice);
	}

	free(h_intermediate);
	cudaFree(d_intermediate);
}
template void d_Sum<float>(float* d_input, float* d_output, size_t n, int batch);
template void d_Sum<double>(double* d_input, double* d_output, size_t n, int batch);
template void d_Sum<int>(int* d_input, int* d_output, size_t n, int batch);


////////////////
//CUDA kernels//
////////////////

//Slightly modified version of the reduce kernel from CUDA SDK 5.5
template <class T, uint blockSize, bool nIsPow2> __global__ void SumKernel(T* d_input, T* d_output, size_t n)
{
    __shared__ T sdata[blockSize];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    uint tid = threadIdx.x;
    size_t i = blockIdx.x * blockSize * 2 + threadIdx.x;
    uint gridSize = blockSize * 2 * gridDim.x;

    T mySum = 0;
	T c = 0, y, t, val;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
		val = d_input[i];
		y = val - c;
		t = mySum + y;
		c = (t - mySum) - y;
		mySum = t;

        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n)
		{
			val = d_input[i + blockSize];
			y = val - c;
			t = mySum + y;
			c = (t - mySum) - y;
			mySum = t;
		}

        i += gridSize;
    }

    // each thread puts its local sum into shared memory
    sdata[tid] = mySum;
    __syncthreads();


    // do reduction in shared mem
    if (blockSize >= 512)
    {
        if (tid < 256)
        {
            sdata[tid] = mySum = mySum + sdata[tid + 256];
        }

        __syncthreads();
    }

    if (blockSize >= 256)
    {
        if (tid < 128)
        {
            sdata[tid] = mySum = mySum + sdata[tid + 128];
        }

        __syncthreads();
    }

    if (blockSize >= 128)
    {
        if (tid <  64)
        {
            sdata[tid] = mySum = mySum + sdata[tid +  64];
        }

        __syncthreads();
    }

    if (tid < 32)
    {
		T* smem = sdata;

        if (blockSize >=  64)
        {
            smem[tid] = mySum = mySum + smem[tid + 32];
			__syncthreads();
        }

        if (blockSize >=  32)
        {
            smem[tid] = mySum = mySum + smem[tid + 16];
			__syncthreads();
        }

        if (blockSize >=  16)
        {
            smem[tid] = mySum = mySum + smem[tid +  8];
			__syncthreads();
        }

        if (blockSize >=   8)
        {
            smem[tid] = mySum = mySum + smem[tid +  4];
			__syncthreads();
        }

        if (blockSize >=   4)
        {
            smem[tid] = mySum = mySum + smem[tid +  2];
			__syncthreads();
        }

        if (blockSize >=   2)
        {
            smem[tid] = mySum = mySum + smem[tid +  1];
			__syncthreads();
        }
    }

    // write result for this block to global mem
    if (tid == 0)
        d_output[blockIdx.x] = sdata[0];
}