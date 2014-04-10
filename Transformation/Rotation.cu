#include "../Prerequisites.cuh"
#include "../Functions.cuh"
#include "../GLMFunctions.cuh"
#include "../DeviceFunctions.cuh"


////////////////////////////
//CUDA kernel declarations//
////////////////////////////

template<int mode> __global__ void RotateKernel(tfloat* d_output, int3 dims, glm::vec3* d_vec);
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
		RotateKernel<0> <<<grid, TpB>>> (d_output, dims, d_vec);
	else if(mode == T_INTERP_CUBIC)		
		RotateKernel<1> <<<grid, TpB>>> (d_output, dims,  d_vec);	
	
	cudaFree(d_vec);
	free(h_vec);

	cudaUnbindTexture(texRotationVolume);
}

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
	d_ConvertTComplexToSplitComplex(d_input, d_real, d_imag, ElementsFFT(dims));

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
		Rotate2DFTKernel<1> <<<grid, (int)TpB>>> (d_output, dims, vecx, vecy);
	else if(mode == T_INTERP_MODE::T_INTERP_CUBIC)
		Rotate2DFTKernel<2> <<<grid, (int)TpB>>> (d_output, dims, vecx, vecy);

	cudaStreamQuery(0);

	cudaUnbindTexture(texRotation2DFTReal);
	cudaUnbindTexture(texRotation2DFTImag);
	cudaFree(d_pitchedimag);
	cudaFree(d_pitchedreal);
	cudaFree(d_imag);
	cudaFree(d_real);
}


////////////////
//CUDA kernels//
////////////////

template<int mode> __global__ void RotateKernel(tfloat* d_output, int3 dims, glm::vec3* d_vec)
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

	/*float radiussq = pos.x * pos.x + pos.y * pos.y;
	if(radiussq >= (float)(dims.x * dims.x / 4))
	{
		(*d_output).x = (tfloat)0;
		(*d_output).y = (tfloat)0;
		return;
	}*/

	bool isnegative = false;
	if(pos.x > 0)
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