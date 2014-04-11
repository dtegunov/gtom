#include "../Prerequisites.cuh"
#include "../Functions.cuh"
#include "../GLMFunctions.cuh"
#include "../DeviceFunctions.cuh"


////////////////////////////
//CUDA kernel declarations//
////////////////////////////

template<int mode> __global__ void Rotate3DKernel(tfloat* d_output, int3 dims, glm::vec3* d_vec);
template<int mode> __global__ void Rotate2DFTKernel(tcomplex* d_output, int3 dims, glm::vec2 vecx, glm::vec2 vecy);


///////////
//Globals//
///////////

texture<tfloat, 3, cudaReadModeElementType> texRotationVolume;
texture<tfloat, 2, cudaReadModeElementType> texRotation2DFTReal;
texture<tfloat, 2, cudaReadModeElementType> texRotation2DFTImag;


////////////////////
//Rotate 3D volume//
////////////////////

void d_Rotate3D(tfloat* d_input, tfloat* d_output, int3 dims, tfloat3* angles, T_INTERP_MODE mode, int batch)
{	
	cudaExtent volumeSize = make_cudaExtent(dims.x, dims.y, dims.z);
	cudaArray *d_inputArray = 0;

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<tfloat>();
	cudaMalloc3DArray(&d_inputArray, &channelDesc, volumeSize); 

	tfloat* d_prefilter;
	if(mode == T_INTERP_CUBIC)
		cudaMalloc((void**)&d_prefilter, Elements(dims) * sizeof(tfloat));
	
	tfloat* d_source;
	if(mode == T_INTERP_CUBIC)
	{
		cudaMemcpy(d_prefilter, d_input, Elements(dims) * sizeof(tfloat), cudaMemcpyDeviceToDevice);
		d_CubicBSplinePrefilter3D(d_prefilter, dims.x * sizeof(tfloat), dims.x, dims.y, dims.z);
		d_source = d_prefilter;
	}
	else
		d_source = d_input;

	cudaMemcpy3DParms copyParams = {0};
	copyParams.srcPtr = make_cudaPitchedPtr((void*)d_source, dims.x * sizeof(tfloat), dims.x, dims.y); 
	copyParams.dstArray = d_inputArray;
	copyParams.extent = volumeSize;
	copyParams.kind = cudaMemcpyDeviceToDevice;
	cudaMemcpy3D(&copyParams);

	d_Rotate3D(d_inputArray, channelDesc, d_output, dims, angles, mode, batch);
	
	if(mode == T_INTERP_CUBIC)
		cudaFree(d_prefilter);
	cudaFreeArray(d_inputArray);
}

void d_Rotate3D(cudaArray* a_input, cudaChannelFormatDesc channelDesc, tfloat* d_output, int3 dims, tfloat3* angles, T_INTERP_MODE mode, int batch)
{
	texRotationVolume.normalized = false;
	texRotationVolume.filterMode = cudaFilterModeLinear;
	texRotationVolume.addressMode[0] = cudaAddressModeClamp;
	texRotationVolume.addressMode[1] = cudaAddressModeClamp;
	texRotationVolume.addressMode[2] = cudaAddressModeClamp;
	cudaBindTextureToArray(texRotationVolume, a_input, channelDesc);

	glm::vec3* h_vec = (glm::vec3*)malloc(batch * 3 * sizeof(glm::vec3));
	for (int b = 0; b < batch; b++)
	{
		glm::mat4 rotationmat = glm::inverse(GetEulerRotation(angles[b]));
	
		glm::vec4 vecX = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f) * rotationmat;
		glm::vec4 vecY = glm::vec4(0.0f, 1.0f, 0.0f, 1.0f) * rotationmat;
		glm::vec4 vecZ = glm::vec4(0.0f, 0.0f, 1.0f, 1.0f) * rotationmat;

		h_vec[b * 3] = glm::vec3(vecX.x, vecX.y, vecX.z);
		h_vec[b * 3 + 1] = glm::vec3(vecY.x, vecY.y, vecY.z);
		h_vec[b * 3 + 2] = glm::vec3(vecZ.x, vecZ.y, vecZ.z);
	}
	glm::vec3* d_vec = (glm::vec3*)CudaMallocFromHostArray(h_vec, batch * 3 * sizeof(glm::vec3));

	int TpB = min(NextMultipleOf(Elements(dims), 32), 256);
	dim3 grid = dim3(min((Elements(dims) + TpB - 1) / TpB, 8192), batch);

	if(mode == T_INTERP_LINEAR)
		Rotate3DKernel<0> <<<grid, TpB>>> (d_output, dims, d_vec);
	else if(mode == T_INTERP_CUBIC)		
		Rotate3DKernel<1> <<<grid, TpB>>> (d_output, dims,  d_vec);	
	
	cudaFree(d_vec);
	free(h_vec);

	cudaUnbindTexture(texRotationVolume);
}


//////////////////////////////
//Rotate 2D in Fourier space//
//////////////////////////////

void d_Rotate2DFT(tcomplex* d_input, tcomplex* d_output, int3 dims, tfloat angle, T_INTERP_MODE mode, int batch)
{
	texRotation2DFTReal.normalized = false;
	texRotation2DFTReal.filterMode = cudaFilterModeLinear;
	texRotation2DFTImag.normalized = false;
	texRotation2DFTImag.filterMode = cudaFilterModeLinear;

	tfloat* d_real;
	cudaMalloc((void**)&d_real, ElementsFFT(dims) * sizeof(tfloat));
	tfloat* d_imag;
	cudaMalloc((void**)&d_imag, ElementsFFT(dims) * sizeof(tfloat));

	for (int b = 0; b < batch; b++)
	{
		d_ConvertTComplexToSplitComplex(d_input + ElementsFFT(dims) * b, d_real, d_imag, ElementsFFT(dims));

		int pitchedwidth = (dims.x / 2 + 1) * sizeof(tfloat);
		tfloat* d_pitchedreal = (tfloat*)CudaMallocAligned2D((dims.x / 2 + 1) * sizeof(tfloat), dims.y, &pitchedwidth);
		for (int y = 0; y < dims.y; y++)
			cudaMemcpy((char*)d_pitchedreal + y * pitchedwidth, 
						d_real + y * (dims.x / 2 + 1), 
						(dims.x / 2 + 1) * sizeof(tfloat), 
						cudaMemcpyDeviceToDevice);
		tfloat* d_pitchedimag = (tfloat*)CudaMallocAligned2D((dims.x / 2 + 1) * sizeof(tfloat), dims.y, &pitchedwidth);
		for (int y = 0; y < dims.y; y++)
			cudaMemcpy((char*)d_pitchedimag + y * pitchedwidth, 
						d_imag + y * (dims.x / 2 + 1), 
						(dims.x / 2 + 1) * sizeof(tfloat), 
						cudaMemcpyDeviceToDevice);

		if(mode == T_INTERP_CUBIC)
		{
			d_CubicBSplinePrefilter2D(d_pitchedreal, pitchedwidth, toInt2(dims.x / 2 + 1, dims.y));
			d_CubicBSplinePrefilter2D(d_pitchedimag, pitchedwidth, toInt2(dims.x / 2 + 1, dims.y));
		}

		cudaChannelFormatDesc descreal = cudaCreateChannelDesc<tfloat>();
		cudaBindTexture2D(NULL, 
							texRotation2DFTReal, 
							d_pitchedreal, 
							descreal, 
							dims.x / 2 + 1, 
							dims.y, 
							pitchedwidth);
		cudaChannelFormatDesc descimag = cudaCreateChannelDesc<tfloat>();
		cudaBindTexture2D(NULL, 
							texRotation2DFTImag, 
							d_pitchedimag, 
							descimag, 
							dims.x / 2 + 1, 
							dims.y, 
							pitchedwidth);
	
		glm::vec2 vecx = glm::vec2(cos(-angle), sin(-angle));
		glm::vec2 vecy = glm::vec2(cos(-angle + (tfloat)PI / (tfloat)2), sin(-angle + (tfloat)PI / (tfloat)2));

		size_t TpB = min(256, NextMultipleOf(dims.x / 2 + 1, 32));
		dim3 grid = dim3((dims.x / 2 + 1 + TpB - 1) / TpB, dims.y, batch);
	
		if(mode == T_INTERP_MODE::T_INTERP_LINEAR)
			Rotate2DFTKernel<1> <<<grid, (int)TpB>>> (d_output + ElementsFFT(dims) * b, dims, vecx, vecy);
		else if(mode == T_INTERP_MODE::T_INTERP_CUBIC)
			Rotate2DFTKernel<2> <<<grid, (int)TpB>>> (d_output + ElementsFFT(dims) * b, dims, vecx, vecy);

		cudaStreamQuery(0);

		cudaUnbindTexture(texRotation2DFTReal);
		cudaUnbindTexture(texRotation2DFTImag);
		cudaFree(d_pitchedimag);
		cudaFree(d_pitchedreal);
	}

	cudaFree(d_imag);
	cudaFree(d_real);
}

void d_Rotate2D(tfloat* d_input, tfloat* d_output, int3 dims, tfloat angle, int batch)
{
	int3 dimspadded = toInt3(dims.x * 2, dims.y * 2, 1);
	tcomplex* d_padded;
	cudaMalloc((void**)&d_padded, ElementsFFT(dimspadded) * batch * sizeof(tcomplex));

	d_Pad(d_input, (tfloat*)d_padded, dims, dimspadded, T_PAD_MODE::T_PAD_MIRROR, (tfloat)0, batch);
	d_RemapFull2FullFFT((tfloat*)d_padded, (tfloat*)d_padded, dimspadded, batch);
	d_FFTR2C((tfloat*)d_padded, d_padded, 2, dimspadded, batch);
	d_RemapHalfFFT2Half(d_padded, d_padded, dimspadded, batch);

	d_Rotate2DFT(d_padded, d_padded, dimspadded, angle, T_INTERP_CUBIC, batch);

	d_RemapHalf2HalfFFT(d_padded, d_padded, dimspadded, batch);
	d_IFFTC2R(d_padded, (tfloat*)d_padded, 2, dimspadded, batch);
	d_RemapFullFFT2Full((tfloat*)d_padded, (tfloat*)d_padded, dimspadded, batch);
	d_Pad((tfloat*)d_padded, d_output, dimspadded, dims, T_PAD_MODE::T_PAD_VALUE, (tfloat)0, batch);

	cudaFree(d_padded);
}


////////////////
//CUDA kernels//
////////////////

template<int mode> __global__ void Rotate3DKernel(tfloat* d_output, int3 dims, glm::vec3* d_vec)
{
	__shared__ glm::vec3 vecx, vecy, vecz;
	__shared__ uint elementsxy, elements, threads;
	__shared__ glm::vec3 center;

	__syncthreads();
	if(threadIdx.x == 0)
	{
		vecx = d_vec[blockIdx.y * 3];
	}
	else if(threadIdx.x == 1)
	{
		vecy = d_vec[blockIdx.y * 3 + 1];
		elementsxy = dims.x * dims.y;
		elements = elementsxy * dims.z;
		threads = blockDim.x * gridDim.x;
	}
	else if(threadIdx.x == 2)
	{
		vecz = d_vec[blockIdx.y * 3 + 2];
		center = glm::vec3((float)(dims.x / 2), (float)(dims.y / 2), (float)(dims.z / 2));
	}
	d_output += Elements(dims) * blockIdx.y;

	__syncthreads();
	
	for (uint id = blockIdx.x * blockDim.x + threadIdx.x; id < elements; id += threads)
	{
		int z = id / elementsxy;
		int y = (id - z * elementsxy) / dims.x;
		int x = id % dims.x;

		glm::vec3 pos = glm::vec3(x, y, z) - center;
		pos = pos.x * vecx + pos.y * vecy + pos.z * vecz + center;
		tfloat value;
		if(pos.x >= -0.00001f && pos.x < (float)dims.x && pos.y >= -0.00001f && pos.y < (float)dims.y && pos.z >= -0.00001f && pos.z < (float)dims.z)
		{
			if(mode == 0)
				value = tex3D(texRotationVolume, pos.x + 0.5f, pos.y + 0.5f, pos.z + 0.5f);
			else if(mode == 1)
				value = cubicTex3D(texRotationVolume, pos.x + 0.5f, pos.y + 0.5f, pos.z + 0.5f);
		}
		else
			value = (tfloat)0;

		//value = blockIdx.y;

		d_output[id] = value;
	}
}

template<int mode> __global__ void Rotate2DFTKernel(tcomplex* d_output, int3 dims, glm::vec2 vecx, glm::vec2 vecy)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= dims.x / 2 + 1)
		return;
	int idy = blockIdx.y;

	d_output += blockIdx.z * ElementsFFT(dims) + getOffset(idx, idy, dims.x / 2 + 1);

	glm::vec2 pos = (vecx * (float)(idx - dims.x / 2)) + (vecy * (float)(idy - dims.y / 2));

	float radiussq = pos.x * pos.x + pos.y * pos.y;
	if(radiussq > (float)((dims.x / 2 - 1) * (dims.x / 2 - 1)))
	{
		(*d_output).x = (tfloat)0;
		(*d_output).y = (tfloat)0;
		return;
	}

	bool isnegative = false;
	if(pos.x > 0.5)
	{
		pos = -pos;
		isnegative = true;
	}

	pos += glm::vec2((float)(dims.x / 2) + 0.5f, (float)(dims.y / 2) + 0.5f);
	
	tfloat valre, valim;
	if(mode == 1)
	{
		valre = tex2D(texRotation2DFTReal, pos.x, pos.y);
		valim = tex2D(texRotation2DFTImag, pos.x, pos.y);
	}
	else
	{
		valre = cubicTex2D(texRotation2DFTReal, pos.x, pos.y);
		valim = cubicTex2D(texRotation2DFTImag, pos.x, pos.y);
	}

	if(isnegative)
		valim = -valim;

	//tfloat valre = tex2D(texRotation2DFTReal, pos.x + (float)(dims.x / 2) + 0.5f, pos.y + (float)(dims.y / 2) + 0.5f);
	//tfloat valim = tex2D(texRotation2DFTImag, pos.x + (float)(dims.x / 2) + 0.5f, pos.y + (float)(dims.y / 2) + 0.5f);

	(*d_output).x = valre;
	(*d_output).y = valim;
}