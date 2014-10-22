#pragma once
#include "cufft.h"
#include "Prerequisites.cuh"


//////////////
//Resolution//
//////////////

enum T_FSC_MODE 
{ 
	T_FSC_THRESHOLD = 0,
	T_FSC_FIRSTMIN = 1
};

//FSC.cu:
void d_FSC(tfloat* d_volume1, tfloat* d_volume2, int3 dimsvolume, tfloat* d_curve, int maxradius, cufftHandle* plan = NULL, int batch = 1);

//LocalFSC.cu:
void d_LocalFSC(tfloat* d_volume1, tfloat* d_volume2, int3 dimsvolume, tfloat* d_resolution, int windowsize, int maxradius, tfloat threshold);

//AnisotropicFSC:
void d_AnisotropicFSC(tcomplex* d_volumeft1, tcomplex* d_volumeft2, int3 dimsvolume, tfloat* d_curve, int maxradius, tfloat3 direction, tfloat coneangle, tfloat falloff, int batch = 1);
void d_AnisotropicFSCMap(tfloat* d_volume1, tfloat* d_volume2, int3 dimsvolume, tfloat* d_map, int2 anglesteps, int maxradius, T_FSC_MODE fscmode, tfloat threshold, cufftHandle* plan, int batch);

//LocalAnisotropicFSC>
void d_LocalAnisotropicFSC(tfloat* d_volume1, tfloat* d_volume2, int3 dimsvolume, tfloat* d_resolution, int windowsize, int maxradius, int2 anglesteps, tfloat threshold);