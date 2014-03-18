#include "../Prerequisites.cuh"
#include "../Functions.cuh"
#include "../GLMFunctions.cuh"
#include "../DeviceFunctions.cuh"


texture<tfloat, 3, cudaReadModeElementType> texRotationVolume;


////////////////////////////
//CUDA kernel declarations//
////////////////////////////

template<int mode> __global__ void RotateKernel(tfloat* d_output, int3 dims, glm::vec3 vecx, glm::vec3 vecy, glm::vec3 vecz);


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
	
	for (int b = 0; b < batch; b++)
	{
		tfloat* d_source;
		if(mode == T_INTERP_CUBIC)
		{
			cudaMemcpy(d_prefilter, d_input + Elements(dims) * b, Elements(dims) * sizeof(tfloat), cudaMemcpyDeviceToDevice);
			d_CubicBSplinePrefilter3D(d_prefilter, dims.x * sizeof(tfloat), dims.x, dims.y, dims.z);
			d_source = d_prefilter;
		}
		else
			d_source = d_input + Elements(dims) * b;

		cudaMemcpy3DParms copyParams = {0};
		copyParams.srcPtr = make_cudaPitchedPtr((void*)d_source, dims.x * sizeof(tfloat), dims.x, dims.y); 
		copyParams.dstArray = d_inputArray;
		copyParams.extent = volumeSize;
		copyParams.kind = cudaMemcpyDeviceToDevice;
		cudaMemcpy3D(&copyParams);

		d_Rotate3D(d_inputArray, channelDesc, d_output + Elements(dims) * b, dims, angles[b], mode);
	}
	
	if(mode == T_INTERP_CUBIC)
		cudaFree(d_prefilter);
	cudaFreeArray(d_inputArray);
}

void d_Rotate3D(cudaArray* a_input, cudaChannelFormatDesc channelDesc, tfloat* d_output, int3 dims, tfloat3 angles, T_INTERP_MODE mode)
{
	texRotationVolume.normalized = false;
	texRotationVolume.filterMode = cudaFilterModeLinear;
	texRotationVolume.addressMode[0] = cudaAddressModeClamp;
	texRotationVolume.addressMode[1] = cudaAddressModeClamp;
	texRotationVolume.addressMode[2] = cudaAddressModeClamp;
	cudaBindTextureToArray(texRotationVolume, a_input, channelDesc);

	glm::mat4 rotationmat = glm::inverse(GetEulerRotation(angles));
	
	glm::vec4 vecX = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f) * rotationmat;
	glm::vec4 vecY = glm::vec4(0.0f, 1.0f, 0.0f, 1.0f) * rotationmat;
	glm::vec4 vecZ = glm::vec4(0.0f, 0.0f, 1.0f, 1.0f) * rotationmat;

	int TpB = min(NextMultipleOf(Elements(dims), 32), 256);
	dim3 grid = dim3(min((Elements(dims) + TpB - 1) / TpB, 8192));

	if(mode == T_INTERP_LINEAR)
		RotateKernel<0> <<<grid, TpB>>> (d_output, dims, 
										 glm::vec3(vecX.x, vecX.y, vecX.z),
										 glm::vec3(vecY.x, vecY.y, vecY.z),
										 glm::vec3(vecZ.x, vecZ.y, vecZ.z));
	else if(mode == T_INTERP_CUBIC)		
		RotateKernel<1> <<<grid, TpB>>> (d_output, dims, 
										 glm::vec3(vecX.x, vecX.y, vecX.z),
										 glm::vec3(vecY.x, vecY.y, vecY.z),
										 glm::vec3(vecZ.x, vecZ.y, vecZ.z));	
	
	cudaUnbindTexture(texRotationVolume);
}


////////////////
//CUDA kernels//
////////////////

template<int mode> __global__ void RotateKernel(tfloat* d_output, int3 dims, glm::vec3 vecx, glm::vec3 vecy, glm::vec3 vecz)
{
	uint elementsxy = dims.x * dims.y;
	uint elements = elementsxy * dims.z;
	uint threads = blockDim.x * gridDim.x;
	glm::vec3 center((float)(dims.x / 2) + 0.5f, (float)(dims.y / 2) + 0.5f, (float)(dims.z / 2) + 0.5f);

	for (uint id = 0; id < elements; id += threads)
	{
		uint z = id / elementsxy;
		uint y = id - z * elementsxy;
		uint x = id % (uint)dims.x;

		glm::vec3 pos = (float)(x - dims.x / 2) * vecx + (float)(y - dims.y / 2) * vecy + (float)(z - dims.z / 2) * vecz + center;
		tfloat value;
		if(pos.x > 0.0f && pos.x < (float)dims.x && pos.y > 0.0f && pos.y < (float)dims.y && pos.z > 0.0f && pos.z < (float)dims.z)
		{
			if(mode == 0)
				value = tex3D(texRotationVolume, pos.x, pos.y, pos.z);
			else if(mode == 1)
				value = cubicTex3D(texRotationVolume, pos.x, pos.y, pos.z);
		}
		else
			value = (tfloat)0;

		d_output[z * elementsxy + y * dims.x + x] = value;
	}
}