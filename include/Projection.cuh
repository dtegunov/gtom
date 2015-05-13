#pragma once
#include "cufft.h"
#include "Prerequisites.cuh"


//////////////
//Projection//
//////////////

//Backward.cu:
void d_ProjBackward(tfloat* d_volume, int3 dimsvolume, tfloat3 offsetfromcenter, tfloat* d_image, int2 dimsimage, tfloat3* h_angles, tfloat2* h_offsets, tfloat2* h_scales, T_INTERP_MODE mode, bool outputzerocentered, int batch);

//Forward.cu:
void d_ProjForward(tfloat* d_volume, tfloat* d_volumepsf, int3 dimsvolume, tfloat* d_projections, tfloat* d_projectionspsf, tfloat3* h_angles, tfloat2* h_shifts, T_INTERP_MODE mode, int batch);
void d_ProjForward(tcomplex* d_volumeft, tfloat* d_volumepsf, int3 dimsvolume, tcomplex* d_projectionsft, tfloat* d_projectionspsf, tfloat3* h_angles, tfloat2* h_shifts, T_INTERP_MODE mode, bool outputzerocentered, int batch);
void d_ProjForwardRaytrace(tfloat* d_volume, int3 dimsvolume, tfloat3 volumeoffset, tfloat* d_projections, int2 dimsproj, tfloat3* h_angles, tfloat2* h_offsets, tfloat2* h_scales, T_INTERP_MODE mode, int supersample, int batch);