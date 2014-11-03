#pragma once
#include "cufft.h"
#include "Prerequisites.cuh"


//////////////
//Projection//
//////////////

//Backward.cu:
void d_ProjBackward(tfloat* d_volume, int3 dimsvolume, int3 offsetfromcenter, tfloat* d_image, int3 dimsimage, tfloat3* h_angles, tfloat2* h_offsets, tfloat2* h_scales, T_INTERP_MODE mode, int batch);

//Forward.cu:
void d_ProjForward(tfloat* d_volume, int3 dimsvolume, tfloat* d_projections, int3 dimsproj, tfloat3* h_angles, short kernelsize, int batch);

//Weighting.cu:
void d_ExactWeighting(tfloat* d_weights, int2 dimsimage, tfloat3* h_angles, int nimages, tfloat maxfreq, bool iszerocentered);