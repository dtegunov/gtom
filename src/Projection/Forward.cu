#include "Prerequisites.cuh"
#include "Angles.cuh"
#include "DeviceFunctions.cuh"
#include "FFT.cuh"
#include "Helper.cuh"

#define SincWindow 16

////////////////////////////
//CUDA kernel declarations//
////////////////////////////

__global__ void GetFFTSliceSincKernel(tcomplex* d_volume, int cubelength, float centervolume, tcomplex* d_slices, glm::mat2x3 rotation);


/////////////////////////////////////////
//Equivalent of TOM's tom_proj3d method//
/////////////////////////////////////////

void d_ProjForward(tfloat* d_volume, int3 dimsvolume, tfloat* d_projections, int3 dimsproj, tfloat3* h_angles, int batch)
{
	tcomplex* d_volumeft;
	cudaMalloc((void**)&d_volumeft, ElementsFFT(dimsvolume) * sizeof(tcomplex));
	tcomplex* d_projft;
	cudaMalloc((void**)&d_projft, ElementsFFT(dimsproj) * batch * sizeof(tcomplex));
	glm::mat2x3* h_matrices = (glm::mat2x3*)malloc(batch * sizeof(glm::mat2x3));

	for (int i = 0; i < batch; i++)
	{
		glm::mat3 r = Matrix3Euler(tfloat3(h_angles[i].x, h_angles[i].y, -h_angles[i].z));
		h_matrices[i] = glm::mat2x3(r[0][0], r[0][1], r[0][2], r[1][0], r[1][1], r[1][2]);
	}

	d_FFTR2C(d_volume, d_volumeft, 3, dimsvolume);
	d_RemapHalfFFT2Half(d_volumeft, d_volumeft, dimsvolume);

	dim3 grid = dim3(dimsproj.x / 2 + 1, dimsproj.y);
	dim3 TpB = dim3(SincWindow, SincWindow);
	for (int b = 0; b < batch; b++)
		GetFFTSliceSincKernel << <grid, TpB >> > (d_volumeft, dimsvolume.x, (float)dimsvolume.x / 2.0f, d_projft + ElementsFFT(dimsproj) * b, h_matrices[b]);

	d_IFFTC2R(d_projft, d_projections, 2, dimsproj, batch);

	free(h_matrices);
	cudaFree(d_projft);
	cudaFree(d_volumeft);
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