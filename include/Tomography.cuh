#pragma once
#include "cufft.h"
#include "Prerequisites.cuh"


//////////////
//Tomography//
//////////////

void d_InterpolateSingleAxisTilt(tcomplex* d_projft, int3 dimsproj, tcomplex* d_interpolated, tfloat* d_weights, tfloat* h_angles, int interpindex, int maxpoints, tfloat interpradius, int limity);