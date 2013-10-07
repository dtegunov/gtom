/*--------------------------------------------------------------------------*\
Copyright (c) 2008-2010, Danny Ruijters. All rights reserved.
http://www.dannyruijters.nl/cubicinterpolation/
This file is part of CUDA Cubic B-Spline Interpolation (CI).

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
*  Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
*  Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
*  Neither the name of the copyright holders nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are
those of the authors and should not be interpreted as representing official
policies, either expressed or implied.

When using this code in a scientific project, please cite one or all of the
following papers:
*  Daniel Ruijters and Philippe Thévenaz,
   GPU Prefilter for Accurate Cubic B-Spline Interpolation, 
   The Computer Journal, vol. 55, no. 1, pp. 15-20, January 2012.
   http://dannyruijters.nl/docs/cudaPrefilter3.pdf
*  Daniel Ruijters, Bart M. ter Haar Romeny, and Paul Suetens,
   Efficient GPU-Based Texture Interpolation using Uniform B-Splines,
   Journal of Graphics Tools, vol. 13, no. 4, pp. 61-69, 2008.
\*--------------------------------------------------------------------------*/

#ifndef _MEMCPY_CUDA_H_
#define _MEMCPY_CUDA_H_

#include <stdio.h>
#include "internal/math_func.cu"
#include "..\Prerequisites.cuh"


//--------------------------------------------------------------------------
// Copy floating point data from and to the GPU
//--------------------------------------------------------------------------

//! Allocate GPU memory and copy a voxel volume from CPU to GPU memory
//! @return the pitched pointer to the GPU copy of the voxel volume
//! @param host  pointer to the voxel volume in CPU (host) memory
//! @param width   volume width in number of voxels
//! @param height  volume height in number of voxels
//! @param depth   volume depth in number of voxels
cudaPitchedPtr CopyVolumeHostToDevice(const float* host, uint width, uint height, uint depth)
{
	cudaPitchedPtr device = {0};
	const cudaExtent extent = make_cudaExtent(width * sizeof(float), height, depth);
	cudaMalloc3D(&device, extent);
	cudaMemcpy3DParms p = {0};
	p.srcPtr = make_cudaPitchedPtr((void*)host, width * sizeof(float), width, height);
	p.dstPtr = device;
	p.extent = extent;
	p.kind = cudaMemcpyHostToDevice;
	cudaMemcpy3D(&p);
	return device;
}

//! Allocate GPU memory and copy a voxel volume from CPU to GPU memory
//! @return the pitched pointer to the GPU copy of the voxel volume
//! @param host  pointer to the voxel volume in CPU (host) memory
//! @param width   volume width in number of voxels
//! @param height  volume height in number of voxels
//! @param depth   volume depth in number of voxels
cudaPitchedPtr CopyVolumeDeviceToDevice(const float* deviceFrom, uint width, uint height, uint depth)
{
	cudaPitchedPtr deviceTo = {0};
	const cudaExtent extent = make_cudaExtent(width * sizeof(float), height, depth);
	cudaMalloc3D(&deviceTo, extent);
	cudaMemcpy3DParms p = {0};
	p.srcPtr = make_cudaPitchedPtr((void*)deviceFrom, width * sizeof(float), width, height);
	p.dstPtr = deviceTo;
	p.extent = extent;
	p.kind = cudaMemcpyHostToDevice;
	cudaMemcpy3D(&p);
	return deviceTo;
}

//! Copy a voxel volume from GPU to CPU memory, and free the GPU memory
//! @param host  pointer to the voxel volume copy in CPU (host) memory
//! @param device  pitched pointer to the voxel volume in GPU (device) memory
//! @param width   volume width in number of voxels
//! @param height  volume height in number of voxels
//! @param depth   volume depth in number of voxels
//! @note The \host CPU memory should be pre-allocated
void CopyVolumeDeviceToHost(float* host, const cudaPitchedPtr device, uint width, uint height, uint depth)
{
	const cudaExtent extent = make_cudaExtent(width * sizeof(float), height, depth);
	cudaMemcpy3DParms p = {0};
	p.srcPtr = device;
	p.dstPtr = make_cudaPitchedPtr((void*)host, width * sizeof(float), width, height);
	p.extent = extent;
	p.kind = cudaMemcpyDeviceToHost;
	cudaMemcpy3D(&p);
	cudaFree(device.ptr);  //free the GPU volume
}

//! Copy a voxel volume from a pitched pointer to a texture
//! @param tex      [output]  pointer to the texture
//! @param texArray [output]  pointer to the texArray
//! @param volume   [input]   pointer to the the pitched voxel volume
//! @param extent   [input]   size (width, height, depth) of the voxel volume
//! @param onDevice [input]   boolean to indicate whether the voxel volume resides in GPU (true) or CPU (false) memory
//! @note When the texArray is not yet allocated, this function will allocate it
template<class T, enum cudaTextureReadMode mode> void CreateTextureFromVolume(
	texture<T, 3, mode>* tex, cudaArray** texArray,
	const cudaPitchedPtr volume, cudaExtent extent, bool onDevice)
{
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
	if (*texArray == 0) CUDA_SAFE_CALL(cudaMalloc3DArray(texArray, &channelDesc, extent));
	// copy data to 3D array
	cudaMemcpy3DParms p = {0};
	p.extent   = extent;
	p.srcPtr   = volume;
	p.dstArray = *texArray;
	p.kind     = onDevice ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
	cudaMemcpy3D(&p);
	// bind array to 3D texture
	cudaBindTextureToArray(*tex, *texArray, channelDesc);
	tex->normalized = false;  //access with absolute texture coordinates
	tex->filterMode = cudaFilterModeLinear;
}

//! Copy a voxel volume from continuous memory to a texture
//! @param tex      [output]  pointer to the texture
//! @param texArray [output]  pointer to the texArray
//! @param volume   [input]   pointer to the continuous memory with the voxel
//! @param extent   [input]   size (width, height, depth) of the voxel volume
//! @param onDevice [input]   boolean to indicate whether the voxel volume resides in GPU (true) or CPU (false) memory
//! @note When the texArray is not yet allocated, this function will allocate it
template<class T, enum cudaTextureReadMode mode> void CreateTextureFromVolume(
	texture<T, 3, mode>* tex, cudaArray** texArray,
	const T* volume, cudaExtent extent, bool onDevice)
{
	cudaPitchedPtr ptr = make_cudaPitchedPtr((void*)volume, extent.width*sizeof(T), extent.width, extent.height);
	CreateTextureFromVolume(tex, texArray, ptr, extent, onDevice);
}

#endif  //_MEMCPY_CUDA_H_