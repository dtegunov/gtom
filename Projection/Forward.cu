#include "../Prerequisites.cuh"
#include "../Functions.cuh"
#include "../GLMFunctions.cuh"
#include "../DeviceFunctions.cuh"

#define GLM_FORCE_RADIANS
#define GLM_FORCE_INLINE
#define GLM_FORCE_CUDA
#include "../glm/glm.hpp"
#include "../glm/gtx/transform.hpp"
#include "../glm/gtc/matrix_transform.hpp"
#include "../glm/gtx/quaternion.hpp"
#include "../glm/gtx/euler_angles.hpp"
#include "../glm/gtc/type_ptr.hpp"


texture<tfloat, 3, cudaReadModeElementType> texForwprojVolume;


////////////////////////////
//CUDA kernel declarations//
////////////////////////////

__global__ void ProjForwardKernel(tfloat* d_projection, tfloat* d_samples, int3 dimsvolume, int3 dimsimage, glm::mat4 rotation, glm::vec3 ray);


/////////////////////////////////////////
//Equivalent of TOM's tom_proj3d method//
/////////////////////////////////////////

void d_ProjForward(tfloat* d_volume, int3 dimsvolume, tfloat* d_image, tfloat* d_samples, int3 dimsimage, tfloat2* h_angles, int batch)
{
	cudaExtent volumeSize = make_cudaExtent(dimsvolume.x, dimsvolume.y, dimsvolume.z);
	cudaArray *d_volumeArray = 0; //for tex

	/*tfloat* d_prefiltvolume;
	cudaMalloc((void**)&d_prefiltvolume, Elements(dimsvolume) * sizeof(tfloat));
	cudaMemcpy(d_prefiltvolume, d_volume, Elements(dimsvolume) * sizeof(tfloat), cudaMemcpyDeviceToDevice);
	d_CubicBSplinePrefilter3D(d_prefiltvolume, dimsvolume.x * sizeof(tfloat), dimsvolume.x, dimsvolume.y, dimsvolume.z);*/

	//initialize the 3D texture with a 3D array
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<tfloat>();
	cudaMalloc3DArray(&d_volumeArray, &channelDesc, volumeSize); 

	//d_CubicBSplinePrefilter3D(d_volume, (int)(dimsvolume.x * sizeof(tfloat)), dimsvolume.x, dimsvolume.y, dimsvolume.z);
	
	//copy d_volumeMem to 3DArray
	cudaMemcpy3DParms copyParams = {0};
	copyParams.srcPtr = make_cudaPitchedPtr((void*)d_volume, dimsvolume.x * sizeof(tfloat), dimsvolume.x, dimsvolume.y); 
	copyParams.dstArray = d_volumeArray;
	copyParams.extent = volumeSize;
	copyParams.kind = cudaMemcpyDeviceToDevice;
	cudaMemcpy3D(&copyParams); 
	//cudaFree(d_prefiltvolume);

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
		glm::vec4 vecForward = glm::vec4(0.0f, 0.0f, 0.5f, 1.0f);

		/*float phi = PI / 2.0f - h_angles[b].x;
		float psi = h_angles[b].x - PI / 2.0f;
		float theta = h_angles[b].y;

		float cosphi = cos(phi), sinphi = sin(phi);
		float cospsi = cos(psi), sinpsi = sin(psi);
		float costheta = cos(theta), sintheta = sin(theta);

		glm::mat4 rotationMat;

		rotationMat[0][0] = cospsi * cosphi - costheta * sinpsi * sinphi;
		rotationMat[1][0] = sinpsi * cosphi + costheta * cospsi * sinphi;
		rotationMat[2][0] = sintheta * sinphi;
		rotationMat[3][0] = 0.0f;
		rotationMat[0][1] = -cospsi * sinphi - costheta * sinpsi * cosphi;
		rotationMat[1][1] = -sinpsi * sinphi + costheta * cospsi * cosphi;
		rotationMat[2][1] = sintheta * cosphi;
		rotationMat[3][1] = 0.0f;
		rotationMat[0][2] = sintheta * sinpsi;
		rotationMat[1][2] = -sintheta * cospsi;
		rotationMat[2][2] = costheta;
		rotationMat[3][2] = 0.0f;
		rotationMat[0][3] = 0.0f;
		rotationMat[1][3] = 0.0f;
		rotationMat[2][3] = 0.0f;
		rotationMat[3][3] = 1.0f;*/

		glm::mat4 rotationMat = GetEulerRotation(h_angles[b]);

		glm::vec4 vecRay4 = vecForward * rotationMat;
		glm::vec3 vecRay3 = glm::vec3(vecRay4.x, vecRay4.y, vecRay4.z);
		ProjForwardKernel <<<grid, TpB>>> (d_image + Elements(dimsimage) * b, d_samples == NULL ? NULL : d_samples + Elements(dimsimage) * b, dimsvolume, dimsimage, rotationMat, vecRay3);
	}

	cudaUnbindTexture(texForwprojVolume);
	cudaFreeArray(d_volumeArray);
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

__global__ void ProjForwardKernel(tfloat* d_projection, tfloat* d_samples, int3 dimsvolume, int3 dimsimage, glm::mat4 rotation, glm::vec3 ray)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= dimsimage.x)
		return;

	glm::vec3 halfVolume = glm::vec3(dimsvolume.x / 2, dimsvolume.y / 2, dimsvolume.z / 2);

	glm::vec4 origin = glm::vec4((float)(idx - dimsimage.x / 2), (float)((int)blockIdx.y - dimsimage.y / 2), (float)(-dimsvolume.z), 1.0f);
	origin = origin * rotation;
	glm::vec3 origin3 = glm::vec3(origin.x, origin.y, origin.z);

	float tnear = 0.0f, tfar = 0.0f;

	if(intersectBox(origin3, 
					ray, 
					-halfVolume - glm::vec3(0.49999f, 0.49999f, 0.49999f), 
					halfVolume - glm::vec3(0.49999f, 0.49999f, 0.49999f), 
					&tnear, 
					&tfar))
	{
		int steps = ceil(tfar - tnear - 0.01f);
		glm::vec3 stepRay = ray * (tfar - tnear) / (float)(steps);

		origin3 += ray * tnear + halfVolume + glm::vec3(0.5f);
		tfloat raysum = (tfloat)0;
		glm::vec3 raypos = glm::vec3(0);
		//if(origin3.x >= 0.0f && origin3.y >= 0.0f && origin3.z >= 0.0f && origin3.x <= dimsvolume.x && origin3.y <= dimsvolume.y && origin3.z <= dimsvolume.z)
			for(int i = 0; i <= steps; i++)
			{
				raypos = origin3 + (stepRay * (float)i);
				raysum += tex3D(texForwprojVolume, raypos.x, raypos.y, raypos.z);
			}
		//else
			//raysum = -111;

		d_projection[blockIdx.y * dimsimage.x + idx] = raysum / 2.5f;
		if(d_samples != NULL)
			d_samples[blockIdx.y * dimsimage.x + idx] = (tfloat)(steps + 1) / 2.5f;
	}
	else
	{
		d_projection[blockIdx.y * dimsimage.x + idx] = (tfloat)0;
		if(d_samples != NULL)
			d_samples[blockIdx.y * dimsimage.x + idx] = (tfloat)0;
	}
}