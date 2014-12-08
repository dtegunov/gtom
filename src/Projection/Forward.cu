#include "Prerequisites.cuh"
#include "Angles.cuh"
#include "CubicInterp.cuh"
#include "DeviceFunctions.cuh"
#include "FFT.cuh"
#include "Helper.cuh"
#include "Transformation.cuh"

#define SincWindow 16

////////////////////////////
//CUDA kernel declarations//
////////////////////////////

__global__ void GetFFTSliceSincKernel(tcomplex* d_volume, int cubelength, float centervolume, tcomplex* d_slices, glm::mat2x3 rotation);
__global__ void GetFFTSliceCubicKernel(cudaTextureObject_t t_volumeRe, cudaTextureObject_t t_volumeIm, int cubelength, float centervolume, tcomplex* d_slices, glm::mat2x3 rotation);
__global__ void IntersectionKernel(float* d_distmin, float* d_distmax, int2 dims, tfloat3 boxmin, tfloat3 boxmax, glm::vec3 invdirection, char3 signs, glm::mat4 transform);
template <bool cubicinterp> __global__ void RaytraceVolumeKernel(cudaTextureObject_t t_volume, int3 dimsvolume, tfloat* d_projection, int2 dimsimage, float* d_distmin, float* d_distmax, glm::vec3 direction, glm::mat4 transform);


/////////////////////////////////////////
//Equivalent of TOM's tom_proj3d method//
/////////////////////////////////////////

void d_ProjForward(tcomplex* d_volumeft, int3 dimsvolume, tcomplex* d_projectionsft, int3 dimsproj, tfloat3* h_angles, T_INTERP_MODE mode, int batch)
{
	glm::mat2x3* h_matrices = (glm::mat2x3*)malloc(batch * sizeof(glm::mat2x3));

	for (int i = 0; i < batch; i++)
	{
		glm::mat3 r = Matrix3Euler(tfloat3(h_angles[i].x, h_angles[i].y, -h_angles[i].z));
		h_matrices[i] = glm::mat2x3(r[0][0], r[0][1], r[0][2], r[1][0], r[1][1], r[1][2]);
	}

	if (mode == T_INTERP_SINC)
	{
		d_RemapHalfFFT2Half(d_volumeft, d_volumeft, dimsvolume);

		dim3 grid = dim3(dimsproj.x / 2 + 1, dimsproj.y);
		dim3 TpB = dim3(SincWindow, SincWindow);
		for (int b = 0; b < batch; b++)
			GetFFTSliceSincKernel << <grid, TpB >> > (d_volumeft, dimsvolume.x, dimsvolume.x / 2, d_projectionsft + ElementsFFT(dimsproj) * b, h_matrices[b]);
	}
	else if (mode == T_INTERP_CUBIC)
	{
		tcomplex* d_symmetric;
		cudaMalloc((void**)&d_symmetric, Elements(dimsvolume) * sizeof(tcomplex));
		d_HermitianSymmetryPad(d_volumeft, d_symmetric, dimsvolume);
		tfloat* d_volumeRe;
		cudaMalloc((void**)&d_volumeRe, Elements(dimsvolume) * sizeof(tfloat));

		d_Re(d_symmetric, d_volumeRe, Elements(dimsvolume));
		d_RemapFullFFT2Full(d_volumeRe, d_volumeRe, dimsvolume);
		d_CubicBSplinePrefilter3D(d_volumeRe, dimsvolume.x * sizeof(tfloat), dimsvolume);
		cudaArray* a_volumeRe;
		cudaTextureObject_t t_volumeRe;
		d_BindTextureTo3DArray(d_volumeRe, a_volumeRe, t_volumeRe, dimsvolume, cudaFilterModeLinear, false);

		d_Im(d_symmetric, d_volumeRe, Elements(dimsvolume));
		d_RemapFullFFT2Full(d_volumeRe, d_volumeRe, dimsvolume);
		d_CubicBSplinePrefilter3D(d_volumeRe, dimsvolume.x * sizeof(tfloat), dimsvolume);
		cudaArray* a_volumeIm;
		cudaTextureObject_t t_volumeIm;
		d_BindTextureTo3DArray(d_volumeRe, a_volumeIm, t_volumeIm, dimsvolume, cudaFilterModeLinear, false);

		dim3 grid = dim3((dimsproj.x / 2 + 1 + 15) / 16, (dimsproj.y + 15) / 16);
		dim3 TpB = dim3(16, 16);
		for (int b = 0; b < batch; b++)
			GetFFTSliceCubicKernel <<<grid, TpB>>> (t_volumeRe, t_volumeIm, dimsvolume.x, dimsvolume.x / 2, d_projectionsft + ElementsFFT(dimsproj) * b, h_matrices[b]);

		cudaDestroyTextureObject(t_volumeIm);
		cudaFreeArray(a_volumeIm);
		cudaDestroyTextureObject(t_volumeRe);
		cudaFreeArray(a_volumeRe);
		cudaFree(d_volumeRe);
		cudaFree(d_symmetric);
	}

	free(h_matrices);
}

void d_ProjForward(tfloat* d_volume, int3 dimsvolume, tfloat* d_projections, int3 dimsproj, tfloat3* h_angles, T_INTERP_MODE mode, int batch)
{
	tcomplex* d_volumeft;
	cudaMalloc((void**)&d_volumeft, ElementsFFT(dimsvolume) * sizeof(tcomplex));
	tcomplex* d_projft;
	cudaMalloc((void**)&d_projft, ElementsFFT(dimsproj) * batch * sizeof(tcomplex));

	d_FFTR2C(d_volume, d_volumeft, 3, dimsvolume);
	d_ProjForward(d_volumeft, dimsvolume, d_projft, dimsproj, h_angles, mode, batch);
	d_IFFTC2R(d_projft, d_projections, 2, dimsproj, batch);

	cudaFree(d_projft);
	cudaFree(d_volumeft);
}

void d_ProjForwardRaytrace(tfloat* d_volume, int3 dimsvolume, tfloat* d_projections, int2 dimsproj, tfloat3* h_angles, tfloat2* h_offsets, tfloat2* h_scales, T_INTERP_MODE mode, int supersample, int batch)
{
	dimsproj = toInt2(dimsproj.x * supersample, dimsproj.y * supersample);
	dimsvolume = toInt3(dimsvolume.x * supersample, dimsvolume.y * supersample, dimsvolume.z * supersample);

	tfloat* d_superproj, *d_supervolume;
	if (supersample > 1)
	{
		cudaMalloc((void**)&d_superproj, Elements2(dimsproj) * batch * sizeof(tfloat));
		cudaMalloc((void**)&d_supervolume, Elements(dimsvolume) * sizeof(tfloat));
		d_Scale(d_volume, d_supervolume, toInt3(dimsvolume.x / supersample, dimsvolume.y / supersample, dimsvolume.z / supersample), dimsvolume, T_INTERP_FOURIER);
	}
	else
	{
		d_superproj = d_projections;
		d_supervolume = d_volume;
	}

	tfloat* d_prefilteredvolume;
	if (mode == T_INTERP_CUBIC)
		cudaMalloc((void**)&d_prefilteredvolume, Elements(dimsvolume) * sizeof(tfloat));

	float* d_distmin, *d_distmax;
	cudaMalloc((void**)&d_distmin, Elements2(dimsproj) * batch * sizeof(float));
	cudaMalloc((void**)&d_distmax, Elements2(dimsproj) * batch * sizeof(float));

	glm::mat4* h_raytransforms = (glm::mat4*)malloc(batch * sizeof(glm::mat4));
	for (int n = 0; n < batch; n++)
		h_raytransforms[n] = Matrix4Translation(tfloat3(dimsvolume.x / 2 + 0.5f, dimsvolume.y / 2 + 0.5f, dimsvolume.z / 2 + 0.5f)) *
							 Matrix4Euler(h_angles[n]) *
							 Matrix4Translation(tfloat3(-h_offsets[n].x * supersample, -h_offsets[n].y * supersample, 0.0f)) *
							 Matrix4Scale(tfloat3(h_scales[n].x, h_scales[n].y, 1.0f)) *
							 Matrix4Translation(tfloat3(-dimsproj.x / 2, -dimsproj.y / 2, 0));

	tfloat3 boxmin = tfloat3(0, 0, 0);
	tfloat3 boxmax = tfloat3(dimsvolume.x,
							 dimsvolume.y,
							 dimsvolume.z);
	for (int n = 0; n < batch; n++)
	{
		int TpB = min(NextMultipleOf(dimsproj.x, 32), 256);
		dim3 grid = dim3((dimsproj.x + TpB - 1) / TpB, dimsproj.y);
		glm::vec3 direction = Matrix3Euler(h_angles[n]) * glm::vec3(0.0f, 0.0f, -1.0f);
		glm::vec3 invdirection = glm::vec3(1.0f / direction.x, 1.0f / direction.y, 1.0f / direction.z);
		char3 signs = make_char3(invdirection.x < 0.0f ? 1 : 0, invdirection.y < 0.0f ? 1 : 0, invdirection.z < 0.0f ? 1 : 0);

		IntersectionKernel << <grid, TpB >> > (d_distmin + Elements2(dimsproj) * n, d_distmax + Elements2(dimsproj) * n, dimsproj, boxmin, boxmax, invdirection, signs, h_raytransforms[n]);
	}

	cudaArray* a_volume;
	cudaTextureObject_t t_volume;

	if (mode == T_INTERP_CUBIC)
	{
		cudaMemcpy(d_prefilteredvolume, d_supervolume, Elements(dimsvolume) * sizeof(tfloat), cudaMemcpyDeviceToDevice);
		d_CubicBSplinePrefilter3D(d_prefilteredvolume, dimsvolume.x * sizeof(tfloat), dimsvolume);
		d_BindTextureTo3DArray(d_prefilteredvolume, a_volume, t_volume, dimsvolume, cudaFilterModeLinear, false);
	}
	else
	{
		d_BindTextureTo3DArray(d_supervolume, a_volume, t_volume, dimsvolume, cudaFilterModeLinear, false);
	}

	dim3 TpB = dim3(16, 16);
	dim3 grid = dim3((dimsproj.x + 15) / 16, (dimsproj.y + 15) / 16);
	for (int n = 0; n < batch; n++)
	{
		glm::vec3 direction = Matrix3Euler(h_angles[n]) * glm::vec3(0.0f, 0.0f, -1.0f);
		if (mode == T_INTERP_CUBIC)
			RaytraceVolumeKernel<true> << <grid, TpB >> > (t_volume,
														   dimsvolume,
														   d_superproj + Elements2(dimsproj) * n,
														   dimsproj,
														   d_distmin + Elements2(dimsproj) * n,
														   d_distmax + Elements2(dimsproj) * n,
														   direction,
														   h_raytransforms[n]);
		else
			RaytraceVolumeKernel<false> << <grid, TpB >> > (t_volume,
															dimsvolume,
															d_superproj + Elements2(dimsproj) * n,
															dimsproj,
															d_distmin + Elements2(dimsproj) * n,
															d_distmax + Elements2(dimsproj) * n,
															direction,
															h_raytransforms[n]);
	}

	cudaDestroyTextureObject(t_volume);
	cudaFreeArray(a_volume);

	if (supersample > 1)
	{
		d_Scale(d_superproj, d_projections, toInt3(dimsproj), toInt3(dimsproj.x / supersample, dimsproj.y / supersample, 1), T_INTERP_FOURIER);
	}

	free(h_raytransforms);
	cudaFree(d_distmax);
	cudaFree(d_distmin);
	if (mode == T_INTERP_CUBIC)
		cudaFree(d_prefilteredvolume);
	if (supersample > 1)
	{
		cudaFree(d_supervolume);
		cudaFree(d_superproj);
	}
}


////////////////
//CUDA kernels//
////////////////

__global__ void GetFFTSliceSincKernel(tcomplex* d_volume, int cubelength, float centervolume, tcomplex* d_slices, glm::mat2x3 rotation)
{
	__shared__ tcomplex s_sums[SincWindow][SincWindow];
	s_sums[threadIdx.y][threadIdx.x] = make_cuComplex(0.0f, 0.0f);

	int idx = blockIdx.x;
	int idy = blockIdx.y;

	int x = cubelength / 2 - idx;
	int y = cubelength - 1 - ((idy + cubelength / 2 - 1) % cubelength);

	d_slices += (cubelength / 2 + 1) * y;

	glm::vec2 localposition = glm::vec2((float)idx - centervolume, (float)idy - centervolume);
	glm::vec3 position = rotation * localposition + centervolume;
	if (position.x < 0.0f || position.x > cubelength - 1 || position.y < 0.0f || position.y > cubelength - 1 || position.z < 0.0f || position.z > cubelength - 1)
	{
		if (threadIdx.y == 0 && threadIdx.x == 0)
			d_slices[x] = make_cuComplex(0.0f, 0.0f);
		return;
	}

	tcomplex sum = make_cuComplex(0.0f, 0.0f);

	int startx = (int)(position.x + 0.5f) - SincWindow / 2;
	int starty = (int)(position.y + 0.5f) - SincWindow / 2;
	int startz = (int)(position.z + 0.5f) - SincWindow / 2;

	float weightxy = sinc((float)((int)threadIdx.y + starty) - position.y) * sinc((float)((int)threadIdx.x + startx) - position.x);
	int xx = ((int)threadIdx.x + startx + cubelength) % cubelength;
	int yy = ((int)threadIdx.y + starty + cubelength) % cubelength;
	char flip = 0;
	if (xx > cubelength / 2)
	{
		xx = cubelength - 1 - xx;
		yy = cubelength - 1 - yy;
		flip = 1;
	}

	d_volume += yy * (cubelength / 2 + 1) + xx;

	for (int sz = 0; sz < SincWindow; sz++)
	{
		int zz = (int)(sz + cubelength + startz) % cubelength;
		if (flip)
			zz = cubelength - 1 - zz;

		tcomplex summand = d_volume[zz * cubelength * (cubelength / 2 + 1)];
		float weight = weightxy * sinc(abs((float)(sz + startz) - position.z));
		sum += make_cuComplex(summand.x * weight, summand.y * (flip ? -weight : weight));
	}
	
	s_sums[threadIdx.y][threadIdx.x] = sum;
	__syncthreads();

	if (threadIdx.x == 0)
	{
		for (char i = 1; i < blockDim.x; i++)
			sum += s_sums[threadIdx.y][i];
		s_sums[threadIdx.y][0] = sum;
	}
	__syncthreads();

	if (threadIdx.y == 0 && threadIdx.x == 0)
	{
		for (char i = 1; i < blockDim.y; i++)
			sum += s_sums[i][0];
		d_slices[x] = sum;
	}
}

__global__ void GetFFTSliceCubicKernel(cudaTextureObject_t t_volumeRe, cudaTextureObject_t t_volumeIm, int cubelength, float centervolume, tcomplex* d_slices, glm::mat2x3 rotation)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	if (idx > cubelength / 2 || idy >= cubelength)
		return;

	int x = cubelength / 2 - idx;
	int y = cubelength - 1 - ((idy + cubelength / 2 - 1) % cubelength);

	glm::vec2 localposition = glm::vec2((float)idx - centervolume, (float)idy - centervolume);
	glm::vec3 position = rotation * localposition + centervolume + 0.5f;
	
	tfloat valRe = cubicTex3D(t_volumeRe, position.x, position.y, position.z);
	tfloat valIm = cubicTex3D(t_volumeIm, position.x, position.y, position.z);

	d_slices[(cubelength / 2 + 1) * y + x] = make_cuComplex(valRe, -valIm);
}

__global__ void IntersectionKernel(float* d_distmin, float* d_distmax, int2 dims, tfloat3 boxmin, tfloat3 boxmax, glm::vec3 invdirection, char3 signs, glm::mat4 transform)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= dims.x)
		return;
	int idy = blockIdx.y;
	d_distmin += idy * dims.x + idx;
	d_distmax += idy * dims.x + idx;

	glm::vec4 origin = transform * glm::vec4((float)idx, (float)idy, 9999.0f, 1.0f);

	float tmin, tmax, tymin, tymax, tzmin, tzmax;

	tmin = ((signs.x ? boxmax.x : boxmin.x) - origin.x) * invdirection.x;
	tmax = ((signs.x ? boxmin.x : boxmax.x) - origin.x) * invdirection.x;
	tymin = ((signs.y ? boxmax.y : boxmin.y) - origin.y) * invdirection.y;
	tymax = ((signs.y ? boxmin.y : boxmax.y) - origin.y) * invdirection.y;
	if ((tmin > tymax) || (tymin > tmax))
	{
		*d_distmin = 0.0f;
		*d_distmax = 0.0f;
		return;
	}
	if (tymin > tmin)
		tmin = tymin;
	if (tymax < tmax)
		tmax = tymax;
	tzmin = ((signs.z ? boxmax.z : boxmin.z) - origin.z) * invdirection.z;
	tzmax = ((signs.z ? boxmin.z : boxmax.z) - origin.z) * invdirection.z;
	if ((tmin > tzmax) || (tzmin > tmax))
	{
		*d_distmin = 0.0f;
		*d_distmax = 0.0f;
		return;
	}
	if (tzmin > tmin)
		tmin = tzmin;
	if (tzmax < tmax)
		tmax = tzmax;

	if (!isnan(tmin) && !isnan(tmax))
	{
		*d_distmin = tmin;
		*d_distmax = tmax;
	}
	else
	{
		*d_distmin = 0.0f;
		*d_distmax = 0.0f;
	}
}

template <bool cubicinterp> __global__ void RaytraceVolumeKernel(cudaTextureObject_t t_volume, int3 dimsvolume, tfloat* d_projection, int2 dimsimage, float* d_distmin, float* d_distmax, glm::vec3 direction, glm::mat4 transform)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= dimsimage.x)
		return;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	if (idy >= dimsimage.y)
		return;

	float distmin = d_distmin[idy * dimsimage.x + idx];
	float distmax = d_distmax[idy * dimsimage.x + idx];
	d_projection += idy * dimsimage.x + idx;

	float pathlength = distmax - distmin;
	ushort steps = ceil(pathlength * 5.0f);
	double sum = 0.0;
	if (steps > 0)
	{
		float steplength = pathlength / (float)steps;
		glm::vec4 origin4 = transform * glm::vec4((float)idx, (float)idy, 9999.0f, 1.0f);
		glm::vec3 origin = glm::vec3(origin4.x, origin4.y, origin4.z);
		distmin += steplength / 2.0f;

		for (ushort i = 0; i < steps; i++)
		{
			glm::vec3 point = (distmin + (float)i * steplength) * direction + origin;
			if (cubicinterp)
				sum += cubicTex3D(t_volume, point.x, point.y, point.z) * steplength;
			else
				sum += tex3D<tfloat>(t_volume, point.x, point.y, point.z) * steplength;
		}
	}

	*d_projection = sum;
}