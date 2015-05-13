//TO BE DONE LATER

//#include "Prerequisites.cuh"
//#include "CTF.cuh"
//#include "FFT.cuh"
//#include "Generics.cuh"
//#include "Helper.cuh"
//#include "Optimization.cuh"
//#include "Transformation.cuh"
//
//
//__global__ void LocalMinMax1DKernel(tfloat* d_input, int dim, int extent, tfloat2* d_min, tfloat2* d_max, uint2* d_minmap, uint2* d_maxmap);
//
//
////////////////////////////////////////////////////////////////////////////
////Find arbitrary number of local peaks and valleys with a defined extent//
////////////////////////////////////////////////////////////////////////////
//
//void d_LocalMinMax1D(tfloat* d_input, uint dim, uchar extent, tfloat2* &d_min, tfloat2* &d_max, uint* &d_offsetmin, uint* &d_offsetmax, uint batch)
//{
//	
//}
//
//
//////////////////
////CUDA kernels//
//////////////////
//
//__device__ inline void Comparator(uint &keyA, uint &keyB)
//{
//	uint t;
//
//	if (keyA > keyB)
//	{
//		t = keyA;
//		keyA = keyB;
//		keyB = t;
//	}
//}
//
//__global__ void LocalMinMax1DKernel(tfloat* d_input, int dim, int extent, tfloat2* d_min, tfloat2* d_max, uint* d_nummin, uint* d_nummax, uint2* d_minmap, uint2* d_maxmap)
//{
//	__shared__ ushort s_keysmin[1536], s_keysmax[1536];
//	__shared__ uint s_nummin, s_nummax;
//	for (uint i = threadIdx.x; i < 1536; i += blockDim.x)
//	{
//		s_keysmin[i] = 65535;
//		s_keysmax[i] = 65535;
//	}
//	if (threadIdx.x == 0)
//	{
//		s_nummin = 0;
//		s_nummax = 0;
//	}
//	d_input += dim * blockIdx.x;
//	d_minmap += blockIdx.x;
//	d_maxmap += blockIdx.x;
//	__syncthreads();
//
//	for (int i = threadIdx.x; i < dim; i += blockDim.x)
//	{
//		tfloat refval = d_input[i];
//		char ismin = true, ismax = true;
//		int start = max(0, i - extent), finish = min(dim, i + extent + 1);
//		for (int w = start; w < finish; w++)
//		{
//			tfloat val = d_input[w];
//			if (val < refval)
//				ismin = false;
//			else if (val > refval)
//				ismax = false;
//		}
//
//		if (ismin == ismax)
//			continue;
//
//		if (ismin)
//		{
//			uint oldindex = atomicInc(&s_nummin, 1536);
//			s_keysmin[oldindex] = i;
//		}
//		else if (ismax)
//		{
//			uint oldindex = atomicInc(&s_nummax, 1536);
//			s_keysmax[oldindex] = i;
//		}
//	}
//
//
//}