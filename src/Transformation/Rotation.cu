#include "Prerequisites.cuh"
#include "Angles.cuh"
#include "CubicInterp.cuh"
#include "DeviceFunctions.cuh"
#include "FFT.cuh"
#include "Generics.cuh"
#include "Helper.cuh"
#include "Transformation.cuh"


////////////////////////////
//CUDA kernel declarations//
////////////////////////////

template<bool cubicinterp> __global__ void Rotate3DKernel(cudaTextureObject_t t_input, tfloat* d_output, int3 dims, glm::mat4 transform);
template<int mode, bool outputzerocentered> __global__ void Rotate2DFTKernel(cudaTextureObject_t t_Re, cudaTextureObject_t t_Im, tcomplex* d_output, int3 dims, glm::mat2 transform, tfloat maxfreq);
template<int mode> __global__ void Rotate3DFTKernel(cudaTextureObject_t t_Re, cudaTextureObject_t t_Im, tcomplex* d_output, int3 dims, glm::mat4 transform);


////////////////////
//Rotate 3D volume//
////////////////////

void d_Rotate3D(tfloat* d_input, tfloat* d_output, int3 dims, tfloat3* h_angles, T_INTERP_MODE mode, int batch)
{
	tfloat* d_temp;
	if (mode == T_INTERP_CUBIC)
		cudaMalloc((void**)&d_temp, Elements(dims) * sizeof(tfloat));

	for (int b = 0; b < batch; b++)
	{
		cudaArray* a_input;
		cudaTextureObject_t t_input;
		if (mode == T_INTERP_LINEAR)
			d_BindTextureTo3DArray(d_input + Elements(dims) * b, a_input, t_input, dims, cudaFilterModeLinear, false);
		else
		{
			cudaMemcpy(d_temp, d_input + Elements(dims) * b, Elements(dims) * sizeof(tfloat), cudaMemcpyDeviceToDevice);
			d_CubicBSplinePrefilter3D(d_temp, dims.x * sizeof(tfloat), dims);
			d_BindTextureTo3DArray(d_temp, a_input, t_input, dims, cudaFilterModeLinear, false);
		}

		glm::mat4 transform = Matrix4Translation(tfloat3(dims.x / 2 + 0.5f, dims.y / 2 + 0.5f, dims.z / 2 + 0.5f)) *
							  glm::transpose(Matrix4Euler(h_angles[b])) *
							  Matrix4Translation(tfloat3(-dims.x / 2, -dims.y / 2, -dims.z / 2));

		dim3 TpB = dim3(16, 16);
		dim3 grid = dim3((dims.x + 15) / 16, (dims.y + 15) / 16, dims.z);

		if (mode == T_INTERP_LINEAR)
			Rotate3DKernel<false> << <grid, TpB >> > (t_input, d_output, dims, transform);
		else if (mode == T_INTERP_CUBIC)
			Rotate3DKernel<true> << <grid, TpB >> > (t_input, d_output, dims, transform);

		cudaDestroyTextureObject(t_input);
		cudaFreeArray(a_input);
	}

	if (mode == T_INTERP_CUBIC)
		cudaFree(d_temp);
}


//////////////////////////////
//Rotate 2D in Fourier space//
//////////////////////////////

void d_Rotate2DFT(tcomplex* d_input, tcomplex* d_output, int3 dims, tfloat* angles, tfloat maxfreq, T_INTERP_MODE mode, bool isoutputzerocentered, int batch)
{
	tfloat* d_real;
	cudaMalloc((void**)&d_real, ElementsFFT(dims) * sizeof(tfloat));
	tfloat* d_imag;
	cudaMalloc((void**)&d_imag, ElementsFFT(dims) * sizeof(tfloat));

	for (int b = 0; b < batch; b++)
	{
		d_ConvertTComplexToSplitComplex(d_input + ElementsFFT(dims) * b, d_real, d_imag, ElementsFFT(dims));

		if(mode == T_INTERP_CUBIC)
		{
			d_CubicBSplinePrefilter2D(d_real, (dims.x / 2 + 1) * sizeof(tfloat), toInt2(dims.x / 2 + 1, dims.y));
			d_CubicBSplinePrefilter2D(d_imag, (dims.x / 2 + 1) * sizeof(tfloat), toInt2(dims.x / 2 + 1, dims.y));
		}

		cudaArray* a_Re;
		cudaArray* a_Im;
		cudaTextureObject_t t_Re, t_Im;
		d_BindTextureToArray(d_real, a_Re, t_Re, toInt2(dims.x / 2 + 1, dims.y), cudaFilterModeLinear, false);
		d_BindTextureToArray(d_imag, a_Im, t_Im, toInt2(dims.x / 2 + 1, dims.y), cudaFilterModeLinear, false);

		d_Rotate2DFT(t_Re, t_Im, d_output + ElementsFFT(dims) * b, dims, angles[b], maxfreq, mode, isoutputzerocentered);

		cudaDestroyTextureObject(t_Re);
		cudaDestroyTextureObject(t_Im);
		cudaFreeArray(a_Re);
		cudaFreeArray(a_Im);
	}

	cudaFree(d_imag);
	cudaFree(d_real);
}

void d_Rotate2DFT(cudaTextureObject_t t_inputRe, cudaTextureObject_t t_inputIm, tcomplex* d_output, int3 dims, tfloat angle, tfloat maxfreq, T_INTERP_MODE mode, bool isoutputzerocentered)
{
		glm::mat2 rotation = Matrix2Rotation(-angle);

		dim3 TpB = dim3(16, 16);
		dim3 grid = dim3((dims.x / 2 + 1 + 15) / 16, (dims.y + 15) / 16);

		if (isoutputzerocentered)
		{
			if (mode == T_INTERP_MODE::T_INTERP_LINEAR)
				Rotate2DFTKernel<0, true> << <grid, TpB >> > (t_inputRe, t_inputIm, d_output, dims, rotation, maxfreq);
			else if (mode == T_INTERP_MODE::T_INTERP_CUBIC)
				Rotate2DFTKernel<1, true> << <grid, TpB >> > (t_inputRe, t_inputIm, d_output, dims, rotation, maxfreq);
		}
		else
		{
			if (mode == T_INTERP_MODE::T_INTERP_LINEAR)
				Rotate2DFTKernel<0, false> << <grid, TpB >> > (t_inputRe, t_inputIm, d_output, dims, rotation, maxfreq);
			else if (mode == T_INTERP_MODE::T_INTERP_CUBIC)
				Rotate2DFTKernel<1, false> << <grid, TpB >> > (t_inputRe, t_inputIm, d_output, dims, rotation, maxfreq);
		}
}

void d_Rotate2D(tfloat* d_input, tfloat* d_output, int3 dims, tfloat* angles, int batch)
{
	int3 dimspadded = toInt3(dims.x * 2, dims.y * 2, 1);
	tcomplex* d_padded;
	cudaMalloc((void**)&d_padded, ElementsFFT(dimspadded) * batch * sizeof(tcomplex));

	d_Pad(d_input, (tfloat*)d_padded, dims, dimspadded, T_PAD_MODE::T_PAD_VALUE, (tfloat)0, batch);
	d_RemapFull2FullFFT((tfloat*)d_padded, (tfloat*)d_padded, dimspadded, batch);
	d_FFTR2C((tfloat*)d_padded, d_padded, 2, dimspadded, batch);
	d_RemapHalfFFT2Half(d_padded, d_padded, dimspadded, batch);

	d_Rotate2DFT(d_padded, d_padded, dimspadded, angles, dimspadded.x / 2, T_INTERP_CUBIC, false, batch);

	//d_RemapHalf2HalfFFT(d_padded, d_padded, dimspadded, batch);
	d_IFFTC2R(d_padded, (tfloat*)d_padded, 2, dimspadded, batch);
	d_RemapFullFFT2Full((tfloat*)d_padded, (tfloat*)d_padded, dimspadded, batch);
	d_Pad((tfloat*)d_padded, d_output, dimspadded, dims, T_PAD_MODE::T_PAD_VALUE, (tfloat)0, batch);

	cudaFree(d_padded);
}


//////////////////////////////
//Rotate 3D in Fourier space//
//////////////////////////////

void d_Rotate3DFT(tcomplex* d_input, tcomplex* d_output, int3 dims, tfloat3* h_angles, T_INTERP_MODE mode, int batch)
{
	int3 dimsfft = toInt3(dims.x / 2 + 1, dims.y, dims.z);
	tfloat* d_tempRe;
	cudaMalloc((void**)&d_tempRe, ElementsFFT(dims) * sizeof(tfloat));
	tfloat* d_tempIm;
	cudaMalloc((void**)&d_tempIm, ElementsFFT(dims) * sizeof(tfloat));

	for (int b = 0; b < batch; b++)
	{
		cudaArray* a_Re, *a_Im;
		cudaTextureObject_t t_Re, t_Im;

		d_ConvertTComplexToSplitComplex(d_input + ElementsFFT(dims) * b, d_tempRe, d_tempIm, ElementsFFT(dims));
		if (mode == T_INTERP_CUBIC)
		{
			d_CubicBSplinePrefilter3D(d_tempRe, dimsfft.x * sizeof(tfloat), dimsfft);
			d_CubicBSplinePrefilter3D(d_tempIm, dimsfft.x * sizeof(tfloat), dimsfft);
		}
		d_BindTextureTo3DArray(d_tempRe, a_Re, t_Re, dimsfft, cudaFilterModeLinear, false);
		d_BindTextureTo3DArray(d_tempIm, a_Im, t_Im, dimsfft, cudaFilterModeLinear, false);

		glm::mat4 transform = Matrix4Translation(tfloat3(dims.x / 2 + 0.5f, dims.y / 2 + 0.5f, dims.z / 2 + 0.5f)) *
							  glm::transpose(Matrix4Euler(h_angles[b])) *
							  Matrix4Translation(tfloat3(-dims.x / 2, -dims.y / 2, -dims.z / 2));

		dim3 TpB = dim3(16, 16);
		dim3 grid = dim3((dims.x / 2 + 1 + 15) / 16, (dims.y + 15) / 16, dims.z);
		if (mode == T_INTERP_LINEAR)
			Rotate3DFTKernel<false> << <grid, TpB >> > (t_Re, t_Im, d_output + ElementsFFT(dims) * b, dims, transform);
		if (mode == T_INTERP_CUBIC)
			Rotate3DFTKernel<true> << <grid, TpB >> > (t_Re, t_Im, d_output + ElementsFFT(dims) * b, dims, transform);

		cudaDestroyTextureObject(t_Re);
		cudaDestroyTextureObject(t_Im);
		cudaFreeArray(a_Re);
		cudaFreeArray(a_Im);
	}

	cudaFree(d_tempRe);
	cudaFree(d_tempIm);
}


////////////////
//CUDA kernels//
////////////////

template<bool cubicinterp> __global__ void Rotate3DKernel(cudaTextureObject_t t_input, tfloat* d_output, int3 dims, glm::mat4 transform)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= dims.x)
		return;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (y >= dims.y)
		return;
	int z = blockIdx.z;

	glm::vec4 pos = transform * glm::vec4(x, y, z, 1);
	tfloat value;
	if(pos.x >= 0.0f && pos.x < (float)dims.x && pos.y >= 0.0f && pos.y < (float)dims.y && pos.z >= 0.0f && pos.z < (float)dims.z)
	{
		if (cubicinterp)
			value = cubicTex3DSimple<tfloat>(t_input, pos.x, pos.y, pos.z);
		else
			value = tex3D<tfloat>(t_input, pos.x, pos.y, pos.z);
	}
	else
		value = (tfloat)0;

	d_output[(z * dims.y + y) * dims.x + x] = value;
}

template<int mode, bool outputzerocentered> __global__ void Rotate2DFTKernel(cudaTextureObject_t t_Re, cudaTextureObject_t t_Im, tcomplex* d_output, int3 dims, glm::mat2 transform, tfloat maxfreq)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx > dims.x / 2)
		return;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	if (idy >= dims.y)
		return;

	int x, y;
	if (outputzerocentered)
	{
		x = idx;
		y = idy;
	}
	else
	{
		x = dims.x / 2 - idx;
		y = dims.y - 1 - ((idy + dims.y / 2 - 1) % dims.y);
	}

	glm::vec2 pos = transform * glm::vec2(idx - dims.x / 2, idy - dims.y / 2);

	if (glm::length(pos) > maxfreq)
	{
		d_output[y * (dims.x / 2 + 1) + x] = make_cuComplex(0.0f, 0.0f);
		return;
	}

	bool isnegative = false;
	if(pos.x > 0.00001f)
	{
		pos = -pos;
		isnegative = true;
	}

	pos += glm::vec2((float)(dims.x / 2) + 0.5f, (float)(dims.y / 2) + 0.5f);
	
	tfloat valre, valim;
	if(mode == 0)
	{
		valre = tex2D<tfloat>(t_Re, pos.x, pos.y);
		valim = tex2D<tfloat>(t_Im, pos.x, pos.y);
	}
	else
	{
		valre = cubicTex2D(t_Re, pos.x, pos.y);
		valim = cubicTex2D(t_Im, pos.x, pos.y);
	}

	if(isnegative)
		valim = -valim;

	d_output[y * (dims.x / 2 + 1) + x] = make_cuComplex(valre, valim);
}

template<int mode> __global__ void Rotate3DFTKernel(cudaTextureObject_t t_Re, cudaTextureObject_t t_Im, tcomplex* d_output, int3 dims, glm::mat4 transform)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx > dims.x / 2)
		return;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	if (idy >= dims.y)
		return;
	int idz = blockIdx.z;

	glm::vec4 pos = transform * glm::vec4(idx, idy, idz, 1);

	float radiussq = pos.x * pos.x + pos.y * pos.y + pos.z * pos.z;
	if (radiussq > (float)((dims.x / 2 - 1) * (dims.x / 2 - 1)))
	{
		d_output[(idz * dims.y + idy) * (dims.x / 2 + 1) + idx] = make_cuComplex(0, 0);
		return;
	}

	bool isnegative = false;
	if (pos.x > 0.0000001f)
	{
		pos = -pos;
		isnegative = true;
	}

	tfloat valre, valim;
	if (mode == 1)
	{
		valre = tex3D<tfloat>(t_Re, pos.x, pos.y, pos.z);
		valim = tex3D<tfloat>(t_Im, pos.x, pos.y, pos.z);
	}
	else
	{
		valre = cubicTex3D(t_Re, pos.x, pos.y, pos.z);
		valim = cubicTex3D(t_Im, pos.x, pos.y, pos.z);
	}

	if (isnegative)
		valim = -valim;

	d_output[(idz * dims.y + idy) * (dims.x / 2 + 1) + idx] = make_cuComplex(valre, valim);
}