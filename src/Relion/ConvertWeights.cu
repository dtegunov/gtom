#include "Prerequisites.cuh"
#include "Helper.cuh"
#include <thrust/sort.h>

namespace gtom
{
	__global__ void ConvertWeightsDenseKernel(tfloat* d_weights, uint elements, uint elementsperpart, uint elementsperclass, uint nclasses, uint nrot, uint ntrans, tfloat* d_pdfrot, const tfloat* __restrict__ d_pdftrans, const tfloat* __restrict__ d_mindiff2);

	void d_rlnConvertWeightsDense(tfloat* d_weights, uint nparticles, uint nclasses, uint nrot, uint ntrans, tfloat* d_pdfrot, tfloat* d_pdftrans, tfloat* d_mindiff2)
	{
		uint elementsperclass = nrot * ntrans;
		uint elementsperpart = nclasses * elementsperclass;
		uint elements = nparticles * nclasses * elementsperclass;
		int TpB = 128;
		dim3 grid = dim3((elements + TpB - 1) / TpB);
		ConvertWeightsDenseKernel << <grid, TpB >> > (d_weights, elements, elementsperpart, elementsperclass, nclasses, nrot, ntrans, d_pdfrot, d_pdftrans, d_mindiff2);
	}

	__global__ void ConvertWeightsDenseKernel(tfloat* d_weights, uint elements, uint elementsperpart, uint elementsperclass, uint nclasses, uint nrot, uint ntrans, tfloat* d_pdfrot, const tfloat* __restrict__ d_pdftrans, const tfloat* __restrict__ d_mindiff2)
	{
		for (uint id = blockIdx.x * blockDim.x + threadIdx.x; id < elements; id += gridDim.x * blockDim.x)
		{
			uint ipart = id / elementsperpart;
			uint iclass = (id % elementsperpart) / elementsperclass;
			uint irot = (id % elementsperclass) / ntrans;
			uint itrans = id % ntrans;

			tfloat pdfrot = d_pdfrot[iclass * nrot + irot];		
			tfloat pdftrans = d_pdftrans[(ipart * nclasses + iclass) * ntrans + itrans];

			tfloat diff2 = d_weights[id];
			diff2 -= d_mindiff2[ipart];

			tfloat weight = pdfrot * pdftrans;
			weight *= exp(-diff2);

			d_weights[id] = weight;
		}
	}

	void d_rlnWeightsSort(tfloat* d_input, uint n)
	{
		thrust::sort(d_input, d_input + n);
	}
}
