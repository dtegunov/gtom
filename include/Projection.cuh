#pragma once
#include "cufft.h"
#include "Prerequisites.cuh"


//////////////
//Projection//
//////////////

//Backward.cu:
void d_ProjBackward(tfloat* d_volume, int3 dimsvolume, tfloat* d_image, int3 dimsimage, tfloat2* h_angles, tfloat* h_weights, T_INTERP_MODE mode, int batch = 1);

//Forward.cu:
void d_ProjForward(tfloat* d_volume, int3 dimsvolume, tfloat* d_image, tfloat* d_samples, int3 dimsimage, tfloat2* h_angles, int batch = 1);
void d_ProjForward2(tfloat* d_volume, int3 dimsvolume, tfloat* d_image, tfloat* d_samples, int3 dimsimage, tfloat2* h_angles, int batch = 1);