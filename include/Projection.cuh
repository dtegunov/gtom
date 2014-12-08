#pragma once
#include "cufft.h"
#include "Prerequisites.cuh"


//////////////
//Projection//
//////////////

//Backward.cu:
void d_ProjBackward(tfloat* d_volume, int3 dimsvolume, tfloat3 offsetfromcenter, tfloat* d_image, int3 dimsimage, tfloat3* h_angles, tfloat2* h_offsets, tfloat2* h_scales, T_INTERP_MODE mode, bool outputzerocentered, int batch);

//Forward.cu:
void d_ProjForward(tfloat* d_volume, int3 dimsvolume, tfloat* d_projections, int3 dimsproj, tfloat3* h_angles, T_INTERP_MODE mode, int batch);
void d_ProjForward(tcomplex* d_volumeft, int3 dimsvolume, tcomplex* d_projectionsft, int3 dimsproj, tfloat3* h_angles, T_INTERP_MODE mode, int batch);
void d_ProjForwardRaytrace(tfloat* d_volume, int3 dimsvolume, tfloat* d_projections, int2 dimsproj, tfloat3* h_angles, tfloat2* h_offsets, tfloat2* h_scales, T_INTERP_MODE mode, int supersample, int batch);

//Weighting.cu:
template <class T> void d_Exact2DWeighting(T* d_weights, int2 dimsimage, int* h_indices, tfloat3* h_angles, int nimages, tfloat maxfreq, bool iszerocentered, int batch = 1);
template <class T> void d_Exact3DWeighting(T* d_weights, int3 dimsvolume, tfloat3* h_angles, int nimages, tfloat maxfreq, bool iszerocentered);