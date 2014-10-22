#pragma once
#include "cufft.h"
#include "Prerequisites.cuh"


///////////
//Masking//
///////////

//SphereMask.cu:
template <class T> void d_SphereMask(T* d_input, T* d_output, int3 size, tfloat* radius, tfloat sigma, tfloat3* center, int batch = 1);

//IrregularSphereMask.cu:
template <class T> void d_IrregularSphereMask(T* d_input, T* d_output, int3 dims, tfloat* radiusmap, int2 anglesteps, tfloat sigma, tfloat3* center, int batch = 1);

//RectangleMask.cu:
template <class T> void d_RectangleMask(T* d_input, T* d_output, int3 size, int3 rectsize, tfloat sigma, int3* center, int batch);

//Remap.cu:
template <class T> void d_Remap(T* d_input, intptr_t* d_map, T* d_output, size_t elementsmapped, size_t elementsoriginal, T defvalue, int batch = 1);
template <class T> void d_RemapReverse(T* d_input, intptr_t* d_map, T* d_output, size_t elementsmapped, size_t elementsdestination, T defvalue, int batch = 1);
template <class T> void Remap(T* h_input, intptr_t* h_map, T* h_output, size_t elementsmapped, size_t elementsoriginal, T defvalue, int batch = 1);
template <class T> void d_MaskSparseToDense(T* d_input, intptr_t** d_mapforward, intptr_t* d_mapbackward, size_t &elementsmapped, size_t elementsoriginal);
template <class T> void MaskSparseToDense(T* h_input, intptr_t** h_mapforward, intptr_t* h_mapbackward, size_t &elementsmapped, size_t elementsoriginal);

//Windows.cu:
void d_HannMask(tfloat* d_input, tfloat* d_output, int3 dims, tfloat* radius, tfloat3* center, int batch = 1);
void d_HammingMask(tfloat* d_input, tfloat* d_output, int3 dims, tfloat* radius, tfloat3* center, int batch = 1);
void d_GaussianMask(tfloat* d_input, tfloat* d_output, int3 dims, tfloat* sigma, tfloat3* center, int batch = 1);