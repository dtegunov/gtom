#include "../Prerequisites.cuh"
#include "../Functions.cuh"


////////////////////////////////////
//Equivalent of tom_os3_alignStack//
////////////////////////////////////

void d_Align3D(tfloat* d_input, tfloat* d_targets, int3 dims, int numtargets, tfloat5* h_params, int* h_membership, tfloat* h_scores, int maxtranslation, tfloat3 maxrotation, tfloat3 rotationsteps, int rotationrefinements, T_ALIGN_MODE mode, int batch)
{
	
}