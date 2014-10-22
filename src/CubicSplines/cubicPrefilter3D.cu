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

#ifndef _3D_CUBIC_BSPLINE_PREFILTER_H_
#define _3D_CUBIC_BSPLINE_PREFILTER_H_

#include <stdio.h>
#include "internal/cubicPrefilter_kernel.cu"
#include "Prerequisites.cuh"

//--------------------------------------------------------------------------
// Global CUDA procedures
//--------------------------------------------------------------------------
template<class floatN>
__global__ void SamplesToCoefficients3DX(
	floatN* volume,		// in-place processing
	uint pitch,			// width in bytes
	uint width,			// width of the volume
	uint height,		// height of the volume
	uint depth)			// depth of the volume
{
	// process lines in x-direction
	const uint y = blockIdx.x * blockDim.x + threadIdx.x;
	const uint z = blockIdx.y * blockDim.y + threadIdx.y;
	const uint startIdx = (z * height + y) * pitch;

	floatN* ptr = (floatN*)((char*)volume + startIdx);
	ConvertToInterpolationCoefficients(ptr, width, sizeof(floatN));
}

template<class floatN>
__global__ void SamplesToCoefficients3DY(
	floatN* volume,		// in-place processing
	uint pitch,			// width in bytes
	uint width,			// width of the volume
	uint height,		// height of the volume
	uint depth)			// depth of the volume
{
	// process lines in y-direction
	const uint x = blockIdx.x * blockDim.x + threadIdx.x;
	const uint z = blockIdx.y * blockDim.y + threadIdx.y;
	const uint startIdx = z * height * pitch;

	floatN* ptr = (floatN*)((char*)volume + startIdx);
	ConvertToInterpolationCoefficients(ptr + x, height, pitch);
}

template<class floatN>
__global__ void SamplesToCoefficients3DZ(
	floatN* volume,		// in-place processing
	uint pitch,			// width in bytes
	uint width,			// width of the volume
	uint height,		// height of the volume
	uint depth)			// depth of the volume
{
	// process lines in z-direction
	const uint x = blockIdx.x * blockDim.x + threadIdx.x;
	const uint y = blockIdx.y * blockDim.y + threadIdx.y;
	const uint startIdx = y * pitch;
	const uint slice = height * pitch;

	floatN* ptr = (floatN*)((char*)volume + startIdx);
	ConvertToInterpolationCoefficients(ptr + x, depth, slice);
}

inline __device__ __host__ uint PowTwoDivider(uint n)
{
	if (n == 0) return 0;
	uint divider = 1;
	while ((n & divider) == 0) divider <<= 1; 
	return divider;
}

//--------------------------------------------------------------------------
// Exported functions
//--------------------------------------------------------------------------

//! Convert the voxel values into cubic b-spline coefficients
//! @param volume  pointer to the voxel volume in GPU (device) memory
//! @param pitch   width in bytes (including padding bytes)
//! @param width   volume width in number of voxels
//! @param height  volume height in number of voxels
//! @param depth   volume depth in number of voxels
template<class floatN> void d_CubicBSplinePrefilter3D(floatN* volume, int pitch, int3 dims)
{
	// Try to determine the optimal block dimensions
	uint dimX = min(min(PowTwoDivider(dims.x), PowTwoDivider(dims.y)), 64);
	uint dimY = min(min(PowTwoDivider(dims.z), PowTwoDivider(dims.y)), 512/dimX);
	dim3 dimBlock(dimX, dimY);

	// Replace the voxel values by the b-spline coefficients
	dim3 dimGridX(dims.y / dimBlock.x, dims.z / dimBlock.y);
	SamplesToCoefficients3DX<floatN><<<dimGridX, dimBlock>>>(volume, pitch, dims.x, dims.y, dims.z);

	dim3 dimGridY(dims.x / dimBlock.x, dims.z / dimBlock.y);
	SamplesToCoefficients3DY<floatN> << <dimGridY, dimBlock >> >(volume, pitch, dims.x, dims.y, dims.z);

	dim3 dimGridZ(dims.x / dimBlock.x, dims.y / dimBlock.y);
	SamplesToCoefficients3DZ<floatN> << <dimGridZ, dimBlock >> >(volume, pitch, dims.x, dims.y, dims.z);
}
template void d_CubicBSplinePrefilter3D<float>(float* volume, int pitch, int3 dims);

#endif  //_3D_CUBIC_BSPLINE_PREFILTER_H_
