#pragma once
#include "cufft.h"
#include "Prerequisites.cuh"


//////////////////
//Transformation//
//////////////////

//Bin.cu:
void d_Bin(tfloat* d_input, tfloat* d_output, int3 dims, int bincount, int batch = 1);

//Coordinates.cu:
void d_Cart2Polar(tfloat* d_input, tfloat* d_output, int2 dims, T_INTERP_MODE interpolation, int batch = 1);
void d_CartAtlas2Polar(tfloat* d_input, tfloat* d_output, tfloat2* d_offsets, int2 atlasdims, int2 dims, T_INTERP_MODE interpolation, int batch);
int2 GetCart2PolarSize(int2 dims);

//Rotate.cu:
void d_Rotate3D(tfloat* d_input, tfloat* d_output, int3 dims, tfloat3* angles, T_INTERP_MODE mode, int batch = 1);
void d_Rotate3D(cudaArray* a_input, cudaChannelFormatDesc channelDesc, tfloat* d_output, int3 dims, tfloat3* angles, T_INTERP_MODE mode, int batch = 1);
void d_Rotate2DFT(tcomplex* d_input, tcomplex* d_output, int3 dims, tfloat angle, T_INTERP_MODE mode, int batch = 1);
void d_Rotate2D(tfloat* d_input, tfloat* d_output, int3 dims, tfloat angle, int batch = 1);
void d_Rotate3DFT(tcomplex* d_input, tcomplex* d_output, int3 dims, tfloat3* angles, T_INTERP_MODE mode, int batch = 1);
void d_Rotate3DFT(cudaArray* a_inputRe, cudaChannelFormatDesc channelDescRe, cudaArray* a_inputIm, cudaChannelFormatDesc channelDescIm, tcomplex* d_output, int3 dims, tfloat3* angles, T_INTERP_MODE mode, int batch = 1);

//Shift.cu:
void d_Shift(tfloat* d_input, tfloat* d_output, int3 dims, tfloat3* delta, cufftHandle* planforw = NULL, cufftHandle* planback = NULL, tcomplex* d_sharedintermediate = NULL, int batch = 1);
void d_Shift(tcomplex* d_input, tcomplex* d_output, int3 dims, tfloat3* delta, bool iszerocentered = false, int batch = 1);

//Scale.cu:
void d_Scale(tfloat* d_input, tfloat* d_output, int3 olddims, int3 newdims, T_INTERP_MODE mode, cufftHandle* planforw = NULL, cufftHandle* planback = NULL, int batch = 1);