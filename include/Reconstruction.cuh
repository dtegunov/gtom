#pragma once
#include "cufft.h"
#include "Prerequisites.cuh"


//////////////////
//Reconstruction//
//////////////////

//ART.cu:
void d_ART(tfloat* d_projections, int3 dimsproj, char* d_masks, tfloat* d_volume, tfloat* d_volumeerrors, int3 dimsvolume, tfloat2* h_angles, int iterations);

//RecFourier.cu:
void d_ReconstructFourier(tfloat* d_projections, int3 dimsproj, tfloat* d_volume, int3 dimsvolume, tfloat2* h_angles);
void d_ReconstructFourierAdd(tcomplex* d_volumeft, tfloat* d_samples, tfloat* d_projections, int3 dimsproj, int3 dimsvolume, tfloat2* h_angles);