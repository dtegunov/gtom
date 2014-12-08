#pragma once
#include "cufft.h"
#include "Prerequisites.cuh"


//////////////////
//Reconstruction//
//////////////////

//RecFourier.cu:
void d_ReconstructFourier(tfloat* d_projections, tfloat* d_weights, CTFParams* h_ctf, int3 dimsproj, tfloat* d_volume, int3 dimsvolume, tfloat3* h_angles);
void d_ReconstructFourierAdd(tcomplex* d_volumeft, tfloat* d_samples, tfloat* d_projections, tfloat* d_weights, CTFParams* h_ctf, int3 dimsproj, int3 dimsvolume, tfloat3* h_angles);

//RecSIRT.cu:
void d_RecSIRT(tfloat* d_volume, tfloat* d_residual, int3 dimsvolume, tfloat3 offsetfromcenter, tfloat* d_image, int2 dimsimage, int nimages, tfloat3* h_angles, tfloat2* h_offsets, tfloat2* h_scales, T_INTERP_MODE mode, int supersample, int iterations, bool outputzerocentered);

//RecWBP.cu:
void d_RecWBP(tfloat* d_volume, int3 dimsvolume, tfloat3 offsetfromcenter, tfloat* d_image, int2 dimsimage, int nimages, tfloat3* h_angles, tfloat2* h_offsets, tfloat2* h_scales, T_INTERP_MODE mode, int supersample, bool outputzerocentered);