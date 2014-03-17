#include "../Prerequisites.cuh"
#include "../Functions.cuh"
#include "../GLMFunctions.cuh"


////////////////////////////
//CUDA kernel declarations//
////////////////////////////

template<int mode> __global__ void RotateKernel(tfloat* d_output, int3 dims, glm::vec3 vecx, glm::vec3 vecy, glm::vec3 vecz);


////////////////////////////////////////
//Equivalent of TOM's tom_shift method//
////////////////////////////////////////

void d_Rotate3D(tfloat* d_input, tfloat* d_output, int3 dims, tfloat3* angles, T_INTERP_MODE mode, int batch)
{
	
}


////////////////
//CUDA kernels//
////////////////
