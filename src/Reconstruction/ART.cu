#include "Prerequisites.cuh"
#include "Angles.cuh"
#include "Generics.cuh"
#include "Helper.cuh"

#define CorrectionTpB 64

texture<tfloat, 2> texArtCorrectionAtlas;


////////////////////////////
//CUDA kernel declarations//
////////////////////////////

__global__ void CorrectionsKernel(tfloat* d_volume, int3 dimsvolume, int3 dimsproj, int atlascolumns, glm::vec3* vecX, glm::vec3* vecY, glm::vec3* vecZ);


////////////////////////////////////////
//Performs 3D reconstruction using ART//
////////////////////////////////////////

//NEEDS TO BE REWRITTEN, start around line 57
/*void d_ART(tfloat* d_projections, int3 dimsproj, char* d_masks, tfloat* d_volume, tfloat* d_volumeerrors, int3 dimsvolume, tfloat3* h_angles, int iterations)
{
	tfloat2* d_angles = (tfloat2*)CudaMallocFromHostArray(h_angles, dimsproj.z * sizeof(tfloat3));
	tfloat* d_forwproj;
	cudaMalloc((void**)&d_forwproj, Elements(dimsproj) * sizeof(tfloat));
	tfloat* d_samples;
	cudaMalloc((void**)&d_samples, Elements(dimsproj) * sizeof(tfloat));
	d_ValueFill(d_volume, Elements(dimsvolume), (tfloat)0);

	glm::mat3x2* h_matrices = (glm::mat3x2*)malloc(dimsproj.z * sizeof(glm::mat3x2));

	for (int b = 0; b < dimsproj.z; b++)
	{
		glm::mat3 r = Matrix3Euler(h_angles[b]);
		h_matrices[b] = glm::mat3x2(r[0][0], r[1][0], r[0][1], r[1][1], r[0][2], r[1][2]);
	}

	glm::mat3x2* d_matrices = (glm::mat3x2*)CudaMallocFromHostArray(h_matrices, dimsproj.z * sizeof(glm::mat3x2));

	for (int i = 0; i < iterations; i++)
	{
		d_ProjForward(d_volume, dimsvolume, d_forwproj, toInt3(dimsproj.x, dimsproj.y, 1), h_angles, dimsproj.z);
		tfloat* h_forwproj = (tfloat*)MallocFromDeviceArray(d_forwproj, Elements(dimsproj) * sizeof(tfloat));
		tfloat* h_samples = (tfloat*)MallocFromDeviceArray(d_samples, Elements(dimsproj) * sizeof(tfloat));
		d_SubtractVector(d_projections, d_forwproj, d_forwproj, Elements(dimsproj));
		int3 atlasdims;
		int2 atlasprimitivesperdim;
		int2* h_atlascoords = (int2*)malloc(dimsproj.z * sizeof(int2));
		tfloat* d_corrections = d_MakeAtlas(d_forwproj, dimsproj, atlasdims, atlasprimitivesperdim, h_atlascoords);
		
		cudaChannelFormatDesc descInput = cudaCreateChannelDesc<tfloat>();
		texArtCorrectionAtlas.normalized = false;
		texArtCorrectionAtlas.filterMode = cudaFilterModeLinear;
		cudaBindTexture2D(0,
						  texArtCorrectionAtlas, 
						  d_corrections, 
						  descInput, 
						  atlasdims.x, 
						  atlasdims.y, 
						  atlasdims.x * sizeof(tfloat));

		size_t TpB = CorrectionTpB;
		dim3 grid = dim3(dimsvolume.x, dimsvolume.y, dimsvolume.z);
		CorrectionsKernel <<<grid, TpB>>> (d_volume, dimsvolume, dimsproj, atlasprimitivesperdim.x, d_vecX, d_vecY, d_vecZ);

		tfloat* h_corrections = (tfloat*)MallocFromDeviceArray(d_corrections, Elements(atlasdims) * sizeof(tfloat));
		tfloat* h_volume = (tfloat*)MallocFromDeviceArray(d_volume, Elements(dimsvolume) * sizeof(tfloat));

		free(h_samples);
		free(h_forwproj);
		free(h_corrections);
		free(h_volume);

		cudaUnbindTexture(texArtCorrectionAtlas);
		cudaFree(d_corrections);
		free(h_atlascoords);
	}
	
	cudaFree(d_vecX);
	cudaFree(d_vecY);
	cudaFree(d_vecZ);
	cudaFree(d_samples);
	cudaFree(d_forwproj);
	cudaFree(d_angles);
}


////////////////
//CUDA kernels//
////////////////

__global__ void CorrectionsKernel(tfloat* d_volume, int3 dimsvolume, int3 dimsproj, int atlascolumns, glm::vec3* d_vecX, glm::vec3* d_vecY, glm::vec3* d_vecZ)
{
	int xvol = blockIdx.x;
	if(xvol >= dimsvolume.x)
		return;

	__shared__ tfloat correctionvals[CorrectionTpB];

	tfloat correctionval = 0;
	int samples = 0;
	for (int b = threadIdx.x; b < dimsproj.z; b += blockDim.x)
	{
		glm::vec3 rotated = (float)(xvol - dimsvolume.x / 2) * d_vecX[b] + (float)((int)blockIdx.y - dimsvolume.y / 2) * d_vecY[b] + (float)((int)blockIdx.z - dimsvolume.z / 2) * d_vecZ[b];

		rotated.x += (tfloat)(dimsproj.x / 2) + (tfloat)0.5;
		rotated.y += (tfloat)(dimsproj.y / 2) + (tfloat)0.5;
		if(rotated.x < (tfloat)0 || rotated.x >= (tfloat)dimsproj.x)
			continue;
		if(rotated.y < (tfloat)0 || rotated.y >= (tfloat)dimsproj.y)
			continue;

		rotated.x += (tfloat)(dimsproj.x * (b % atlascolumns));
		rotated.y += (tfloat)(dimsproj.x * (b / atlascolumns));

		correctionval += tex2D(texArtCorrectionAtlas, 
								rotated.x, 
								rotated.y);
	}
	correctionvals[threadIdx.x] = correctionval;

	__syncthreads();

	if(threadIdx.x != 0)
		return;

	for (int i = 1; i < CorrectionTpB; i++)
	{
		correctionval += correctionvals[i];
	}

	d_volume[(blockIdx.z * dimsvolume.y + blockIdx.y) * dimsvolume.x + xvol] += correctionval / (tfloat)dimsproj.z;
}*/