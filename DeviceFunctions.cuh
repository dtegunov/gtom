#include "Prerequisites.cuh"

__device__ float cubicTex1D(texture<float, cudaTextureType1D> tex, float x);
__device__ float cubicTex2D(texture<float, cudaTextureType2D> tex, float x, float y);
__device__ float cubicTex3D(texture<float, cudaTextureType3D> tex, float x, float y, float z);