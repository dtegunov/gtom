#include "../Prerequisites.cuh"
#include "../Functions.cuh"

texture<tfloat, 3, cudaReadModeElementType> texForwprojVolume;


////////////////////////////
//CUDA kernel declarations//
////////////////////////////

__global__ void ProjForwardKernel(tfloat* d_projection, int3 dimsvolume, int3 dimsimage, glm::vec3 camera, glm::vec3 pixelX, glm::vec3 pixelY, glm::vec3 ray);


/////////////////////////////////////////
//Equivalent of TOM's tom_proj3d method//
/////////////////////////////////////////

void d_ProjForward(tfloat* d_volume, int3 dimsvolume, tfloat* d_image, int3 dimsimage, tfloat2* angles, int batch)
{
	cudaExtent volumeSize = make_cudaExtent(dimsvolume.x, dimsvolume.y, dimsvolume.z);
	cudaArray *d_volumeArray = 0; //for tex

	//initialize the 3D texture with a 3D array
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<tfloat>();
	cudaMalloc3DArray(&d_volumeArray, &channelDesc, volumeSize); 
	
	//copy d_volumeMem to 3DArray
	cudaMemcpy3DParms copyParams = {0};
	copyParams.srcPtr = make_cudaPitchedPtr((void*)d_volume, dimsvolume.x * sizeof(tfloat), dimsvolume.x, dimsvolume.y); 
	copyParams.dstArray = d_volumeArray;
	copyParams.extent = volumeSize;
	copyParams.kind = cudaMemcpyDeviceToDevice;
	cudaMemcpy3D(&copyParams); 

	// set texture parameters
	texForwprojVolume.normalized = false;
	texForwprojVolume.filterMode = cudaFilterModeLinear;
	texForwprojVolume.addressMode[0] = cudaAddressModeClamp;
	texForwprojVolume.addressMode[1] = cudaAddressModeClamp;
	texForwprojVolume.addressMode[2] = cudaAddressModeClamp;
	cudaBindTextureToArray(texForwprojVolume, d_volumeArray, channelDesc);

	size_t TpB = min(192, NextMultipleOf(dimsimage.x, 32));
	dim3 grid = dim3((dimsimage.x + TpB - 1) / TpB, dimsimage.y, 1);
	for (int b = 0; b < batch; b++)
	{
		glm::vec4 vecForward = glm::vec4(0.0f, 0.0f, 1.0f, 1.0f);
		glm::vec4 vecBackward = glm::vec4(0.0f, 0.0f, -(dimsvolume.z * 2), 1.0f);
		glm::vec4 vecX = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
		glm::vec4 vecY = glm::vec4(0.0f, 1.0f, 0.0f, 1.0f);

		glm::mat4 rotationMat = glm::mat4(1.0f);

		tfloat cphi = cos(angles[b].x);
		tfloat sphi = sin(angles[b].x);
		tfloat cthe = cos(angles[b].y);
		tfloat sthe = sin(angles[b].y);

		float* matvalues = (float*)glm::value_ptr(rotationMat);
		matvalues[0] = cthe * cphi * cphi + sphi * sphi;
		matvalues[4] = cthe * cphi * sphi - cphi * sphi;
		matvalues[7] = -sthe * cphi;

		matvalues[1] = matvalues[4];
		matvalues[5] = cthe * sphi * sphi + cphi * cphi;
		matvalues[9] = -sthe * sphi;

		matvalues[2] = -matvalues[8];
		matvalues[6] = -matvalues[9];
		matvalues[10] = cthe;

		glm::vec4 vecCamera4 = vecBackward * rotationMat;
		glm::vec3 vecCamera3 = glm::vec3(vecCamera4.x, vecCamera4.y, vecCamera4.z);
		glm::vec4 vecPixelX4 = vecX * rotationMat;
		glm::vec3 vecPixelX3 = glm::vec3(vecPixelX4.x, vecPixelX4.y, vecPixelX4.z);
		glm::vec4 vecPixelY4 = vecY * rotationMat;
		glm::vec3 vecPixelY3 = glm::vec3(vecPixelY4.x, vecPixelY4.y, vecPixelY4.z);
		glm::vec4 vecRay4 = vecForward * rotationMat;
		glm::vec3 vecRay3 = glm::vec3(vecRay4.x, vecRay4.y, vecRay4.z);
		ProjForwardKernel <<<grid, TpB>>> (d_image + Elements(dimsimage) * b, dimsvolume, dimsimage, vecCamera3, vecPixelX3, vecPixelY3, vecRay3);
	}

	cudaUnbindTexture(texForwprojVolume);
}


////////////////
//CUDA kernels//
////////////////

__device__ bool intersectBox(glm::vec3 origin, glm::vec3 ray, glm::vec3 boxmin, glm::vec3 boxmax, float *tnear, float *tfar)
{
    // compute intersection of ray with all six bbox planes
	glm::vec3 invR = glm::vec3(1.0f / ray.x, 1.0f / ray.y, 1.0f / ray.z);
    glm::vec3 tbot = invR * (boxmin - origin);
    glm::vec3 ttop = invR * (boxmax - origin);

    // re-order intersections to find smallest and largest on each axis
    glm::vec3 tmin = glm::vec3(min(tbot.x, ttop.x), min(tbot.y, ttop.y), min(tbot.z, ttop.z));
    glm::vec3 tmax = glm::vec3(max(tbot.x, ttop.x), max(tbot.y, ttop.y), max(tbot.z, ttop.z));

    // find the largest tmin and the smallest tmax
    float largest_tmin = max(max(tmin.x, tmin.y), max(tmin.x, tmin.z));
    float smallest_tmax = min(min(tmax.x, tmax.y), min(tmax.x, tmax.z));

    *tnear = largest_tmin;
    *tfar = smallest_tmax;

    return smallest_tmax >= largest_tmin;
}

__global__ void ProjForwardKernel(tfloat* d_projection, int3 dimsvolume, int3 dimsimage, glm::vec3 camera, glm::vec3 pixelX, glm::vec3 pixelY, glm::vec3 ray)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= dimsimage.x)
		return;

	glm::vec3 halfVolume = glm::vec3(dimsvolume.x / 2, dimsvolume.y / 2, dimsvolume.z / 2);
	//glm::vec3 halfImage = glm::vec3(dimsimage.x / 2, dimsimage.y / 2, 0);

	glm::vec3 origin = camera + ((float)(idx - dimsimage.x / 2) * pixelX) + ((float)((int)blockIdx.y - dimsimage.y / 2) * pixelY);
	float tnear = 0.0f, tfar = 0.0f;

	if(intersectBox(origin, ray, -halfVolume - glm::vec3(0.000001f, 0.000001f, 0.000001f), halfVolume, &tnear, &tfar))
	{
		int steps = ceil(tfar - tnear - 0.00001f);
		glm::vec3 stepRay = ray * (tfar - tnear) / (float)steps;

		origin += ray * tnear + halfVolume + glm::vec3(0.5f);
		tfloat raysum = (tfloat)0;
		glm::vec3 raypos = glm::vec3(0);
		if(origin.x >= 0.0f && origin.y >= 0.0f && origin.z >= 0.0f && origin.x <= dimsvolume.x && origin.y <= dimsvolume.y && origin.z <= dimsvolume.z)
			for(int i = 0; i < steps; i++)
			{
				raypos = origin + (stepRay * (float)i);
				raysum += tex3D(texForwprojVolume, raypos.x, raypos.y, raypos.z);
			}

		d_projection[blockIdx.y * dimsimage.x + idx] = raysum;
	}
	else
		d_projection[blockIdx.y * dimsimage.x + idx] = 0;
}