#include "../Prerequisites.cuh"
#include "../Functions.cuh"
#include <vector>


////////////////////////////
//CUDA kernel declarations//
////////////////////////////

__global__ void LocalPeaksKernel(tfloat* d_input, char* d_output, int3 dims, int localextent, tfloat threshold);


///////////////////////////////////////
//Equivalent of TOM's tom_peak method//
///////////////////////////////////////

void d_LocalPeaks(tfloat* d_input, int3** h_peaks, int* h_peaksnum, int3 dims, int localextent, tfloat threshold, int batch)
{
	size_t TpB = min(32, dims.x);
	size_t blocksx = min((dims.x + TpB - 1) / TpB, 32768);
	dim3 grid = dim3((uint)blocksx, dims.y, dims.z);

	char* h_output;
	cudaMallocHost((void**)&h_output, Elements(dims) * sizeof(char));
	
	vector<int3> peaks;

	for (int b = 0; b < batch; b++)
	{
		peaks.clear();

		char* d_output = CudaMallocValueFilled(Elements(dims), (char)0);

		LocalPeaksKernel <<<grid, (uint)TpB>>> (d_input + Elements(dims) * b, d_output, dims, localextent, threshold);
		cudaMemcpy(h_output, d_output, Elements(dims) * sizeof(char), cudaMemcpyDeviceToHost);
		cudaFree(d_output);

		for (int z = 0; z < dims.z; z++)
			for (int y = 0; y < dims.y; y++)
				for (int x = 0; x < dims.x; x++)
					if(h_output[(z * dims.y + y) * dims.x + x])
						peaks.push_back(toInt3(x, y, z));

		if(peaks.size() > 0)
		{
			h_peaks[b] = (int3*)malloc(peaks.size() * sizeof(int3));
			memcpy(h_peaks[b], &peaks[0], peaks.size() * sizeof(int3));
		}
		h_peaksnum[b] = peaks.size();
	}

	cudaFreeHost(h_output);
}


////////////////
//CUDA kernels//
////////////////

__global__ void LocalPeaksKernel(tfloat* d_input, char* d_output, int3 dims, int localextent, tfloat threshold)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;	
	if(idx >= dims.x)
		return;

	int idy = blockIdx.y;
	int idz = blockIdx.z;

	tfloat value = d_input[(idz * dims.y + idy) * dims.x + idx];
	if(value < threshold)
		return;

	int limx = min(dims.x, idx + localextent);
	int limy = min(dims.y, idy + localextent);
	int limz = min(dims.z, idz + localextent);

	int sqlocalextent = localextent * localextent;
	int sqy, sqz;
	int sqdist;

	for(int z = max(0, idz - localextent); z < limz; z++)
	{
		sqz = idz - z;
		sqz *= sqz;
		for (int y = max(0, idy - localextent); y < limy; y++)
		{
			sqy = idy - y;
			sqy *= sqy;
			sqy += sqz;
			for (int x = max(0, idx - localextent); x < limx; x++)
			{
				sqdist = idx - x;
				sqdist *= sqdist;
				sqdist += sqy;

				if(sqdist > sqlocalextent || sqdist == 0)
					continue;

				if(value <= d_input[(z * dims.y + y) * dims.x + x])
					return;
			}
		}
	}

	d_output[(idz * dims.y + idy) * dims.x + idx] = (char)1;
}