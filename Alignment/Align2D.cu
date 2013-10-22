#include "../Prerequisites.cuh"
#include "../Functions.cuh"


////////////////////////////////////
//Equivalent of tom_os3_alignStack//
////////////////////////////////////

void AlignStack2D(tfloat* h_in_stack, 
				  int3 in_stackdims,
				  tfloat* h_in_refs, 
				  int in_refcount, 
				  char* h_in_mask,
				  int2 in_binning,
				  T_NORM_MODE in_norm,
				  int in_refinements,
				  int in_iterations,
				  char* h_in_masktrans,
				  char* h_in_maskrot,
				  tfloat* h_out_aligned,
				  tfloat* h_out_tmpl,
				  tfloat* h_out_cc,
				  int* h_out_tmplnr,
				  tfloat3* h_out_params)
{
	int3 dims = toInt3(in_stackdims.x, in_stackdims.y, 1);
	int3 dimsPolar = toInt3(GetCart2PolarSize(toInt2(dims.x, dims.y)));
	size_t elements = Elements(dims);
	size_t elementsPolar = Elements(dimsPolar);
	size_t elementsFFT = ElementsFFT(dims);
	size_t elementsPolarFFT = ElementsFFT(dimsPolar);

	tfloat* d_currentstack;
	tfloat* d_refs = (tfloat*)CudaMallocFromHostArray(h_in_refs, elements * in_refcount * sizeof(tfloat));
	char* d_mask = NULL;
	if(h_in_mask != NULL)
		d_mask = (char*)CudaMallocFromHostArray(h_in_mask, elements * sizeof(char));
	char* d_masktrans = NULL;
	if(h_in_masktrans != NULL)
		d_masktrans = (char*)CudaMallocFromHostArray(h_in_masktrans, elements * sizeof(char));
	char* d_maskrot = NULL;
	if(h_in_maskrot != NULL)
		d_maskrot = (char*)CudaMallocFromHostArray(h_in_maskrot, elementsPolar * sizeof(char));

	tfloat* d_aligned;
	tfloat* d_tmpl;
	tfloat* d_cc;
	tfloat3* d_params;
	int* d_tmplnr;
}