#include "../Prerequisites.cuh"
#include "../Functions.cuh"
#include "../GLMFunctions.cuh"
#include "../DeviceFunctions.cuh"


texture<tfloat, 3, cudaReadModeElementType> texRotationVolume;


////////////////////////////
//CUDA kernel declarations//
////////////////////////////

template<int mode> __global__ void RotateKernel(tfloat* d_output, int3 dims, glm::vec3* d_vec);


////////////////////////////////////////
//Equivalent of TOM's tom_shift method//
////////////////////////////////////////

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