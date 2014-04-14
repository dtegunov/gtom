#include "../Prerequisites.cuh"
#include "../Functions.cuh"


////////////////////////////
//CUDA kernel declarations//
////////////////////////////

__global__ void InterpolateSingleAxisTiltKernel(tcomplex* d_projft, size_t elementsproj, tcomplex* d_interpolated, tfloat* d_factors, short* d_indices, short npoints);


////////////////////////////////////////
//Performs 3D reconstruction using ART//
////////////////////////////////////////

void d_InterpolateSingleAxisTilt(tcomplex* d_projft, int3 dimsproj, tcomplex* d_interpolated, tfloat* h_angles, int interpindex, int maxpoints, tfloat smoothsigma)
{
	tfloat interpangle = h_angles[interpindex];
	int npoints = maxpoints;// dimsproj.z - 1;
	tfloat* h_factors = (tfloat*)malloc(npoints * sizeof(tfloat));
	short* h_indices = (short*)malloc(npoints * sizeof(short));

	int n = 0;
	for (int i = max(0, interpindex - maxpoints / 2); i < min(dimsproj.z, interpindex + maxpoints / 2 + 1); i++)
	{
		if(i == interpindex)
			continue;

		double factor = 1.0;
		for (int j = max(0, interpindex - maxpoints / 2); j < min(dimsproj.z, interpindex + maxpoints / 2 + 1); j++)
		{
			if(j == interpindex || j == i)
				continue;
			factor *= ((double)interpangle - (double)h_angles[j]) / ((double)h_angles[i] - (double)h_angles[j]);
		}
		h_factors[n] = (tfloat)factor;

		h_indices[n] = (short)i;
		n++;
	}

	tfloat* d_factors = (tfloat*)CudaMallocFromHostArray(h_factors, n * sizeof(tfloat));
	short* d_indices = (short*)CudaMallocFromHostArray(h_indices, n * sizeof(short));
	free(h_factors);
	free(h_indices);

	int TpB = min(NextMultipleOf((dimsproj.x / 2 + 1) * dimsproj.y, 32), 128);
	dim3 grid = dim3(min(((dimsproj.x / 2 + 1) * dimsproj.y + TpB - 1) / TpB, 8192));
	InterpolateSingleAxisTiltKernel <<<grid, TpB>>> (d_projft, (dimsproj.x / 2 + 1) * dimsproj.y, d_interpolated, d_factors, d_indices, n);

	cudaFree(d_factors);
	cudaFree(d_indices);
}


////////////////
//CUDA kernels//
////////////////

__global__ void InterpolateSingleAxisTiltKernel(tcomplex* d_projft, size_t elementsproj, tcomplex* d_interpolated, tfloat* d_factors, short* d_indices, short npoints)
{
	for(size_t id = blockIdx.x * blockDim.x + threadIdx.x; 
		id < elementsproj; 
		id += blockDim.x * gridDim.x)
	{
		double sumre = 0.0;
		double sumim = 0.0;
		for (int n = 0; n < npoints; n++)
		{
			size_t index = d_indices[n];
			double factor = (double)d_factors[n];
			sumre += (double)d_projft[index * elementsproj + id].x * factor;
			sumim += (double)d_projft[index * elementsproj + id].y * factor;
		}
		d_interpolated[id].x = (tfloat)sumre;
		d_interpolated[id].y = (tfloat)sumim;
	}
}