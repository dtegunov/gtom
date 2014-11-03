#include "Prerequisites.cuh"
#include "Angles.cuh"
#include "DeviceFunctions.cuh"
#include "FFT.cuh"
#include "Helper.cuh"


////////////////////////////
//CUDA kernel declarations//
////////////////////////////

template <class T> __global__ void GetFFTSliceSincKernel(tcomplex* d_volume, ushort cubelength, float centervolume, tcomplex* d_slices, glm::mat2x3 rotation, short windowsize);


/////////////////////////////////////////
//Equivalent of TOM's tom_proj3d method//
/////////////////////////////////////////

void d_ProjForward(tfloat* d_volume, int3 dimsvolume, tfloat* d_projections, int3 dimsproj, tfloat3* h_angles, short kernelsize, int batch)
{
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

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

	dim3 grid = dim3((dimsproj.x / 2 + 1 + 127) / 128, dimsproj.y);
	dim3 TpB = dim3(128);
	for (int b = 0; b < batch; b++)
		GetFFTSliceSincKernel <float> <<<grid, TpB>>> (d_volumeft, dimsvolume.x, (float)dimsvolume.x / 2.0f, d_projft + ElementsFFT(dimsproj) * b, h_matrices[b], kernelsize);

	d_RemapHalf2HalfFFT(d_projft, d_projft, dimsproj, batch);
	d_IFFTC2R(d_projft, d_projections, 2, dimsproj, batch);

	free(h_matrices);
	cudaFree(d_projft);
	cudaFree(d_volumeft);

	cudaDeviceSetCacheConfig(cudaFuncCachePreferNone);
}


////////////////
//CUDA kernels//
////////////////

template <class T> __global__ void GetFFTSliceSincKernel(tcomplex* d_volume, ushort cubelength, float centervolume, tcomplex* d_slices, glm::mat2x3 rotation, short windowsize)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx > cubelength / 2)
		return;
	int idy = blockIdx.y;

	d_slices += (cubelength / 2 + 1) * idy;

	glm::vec2 localposition = glm::vec2((float)idx - centervolume, (float)blockIdx.y - centervolume);
	glm::vec3 position = rotation * localposition + centervolume;
	if (position.x < 0.0f || position.x > cubelength - 1 || position.y < 0.0f || position.y > cubelength - 1 || position.z < 0.0f || position.z > cubelength - 1)
	{
		d_slices[idx] = make_cuComplex(0.0f, 0.0f);
		return;
	}

	short startx = (short)(position.x) - windowsize;
	short starty = (short)(position.y) - windowsize;
	short startz = (short)(position.z) - windowsize;
	windowsize *= 2;

	short scubelength = (short)cubelength;
	float interpRe = 0.0f, interpIm = 0.0f;
	float cRe = 0, yRe, tRe;
	float cIm = 0, yIm, tIm;

	for (short z = 0; z <= windowsize; z++)
	{
		T weightz = sinc(abs((T)(z + startz) - (T)position.z));
		ushort zz = (ushort)(z + scubelength + startz) % cubelength;

		for (short y = 0; y <= windowsize; y++)
		{
			T weighty = sinc(abs((T)(y + starty) - (T)position.y));
			ushort yy = (ushort)(y + scubelength + starty) % cubelength;

			for (short x = 0; x <= windowsize; x++)
			{
				ushort xx = (ushort)(x + scubelength + startx) % cubelength;

				char flip = 0;
				if (xx > cubelength / 2)
				{
					xx = cubelength - 1 - xx;
					flip = 1;
				}

				tcomplex summand = d_volume[((flip ? cubelength - 1 - zz : zz) * cubelength + (flip ? cubelength - 1 - yy : yy)) * (cubelength / 2 + 1) + xx];

				T weight = sinc(abs((T)(x + startx) - (T)position.x)) * weighty * weightz;
				yRe = summand.x * weight - cRe;
				yIm = (flip ? -summand.y : summand.y) * weight - cIm;
				tRe = interpRe + yRe;
				tIm = interpIm + yIm;
				cRe = (tRe - interpRe) - yRe;
				cIm = (tIm - interpIm) - yIm;
				interpRe = tRe;
				interpIm = tIm;
				//interpRe += summand.x * weight;
				//interpIm += (flip ? -summand.y : summand.y) * weight;
			}
		}
	}

	d_slices[idx] = make_cuComplex(interpRe, interpIm);
}