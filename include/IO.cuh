#pragma once
#include "cufft.h"
#include "Prerequisites.cuh"


//////
//IO//
//////

 //mrc.cu:
void ReadMRC(string path, void** data, EM_DATATYPE &datatype, int nframe = 0, bool flipx = false);
void ReadMRCDims(string path, int3 &dims);

//em.cu:
void ReadEM(string path, void** data, EM_DATATYPE &datatype, int nframe = 0);
void ReadEMDims(string path, int3 &dims);

//raw.cu:
void ReadRAW(string path, void** data, EM_DATATYPE datatype, int3 dims, int nframe = 0, size_t headerbytes = 0);
