#include "Prerequisites.cuh"
#include "FFT.cuh"
#include "Helper.cuh"
#include "Generics.cuh"
#include "Masking.cuh"
#include "Resolution.cuh"


///////////////////////////
//CUDA kernel declaration//
///////////////////////////



///////////////////////////////////////////////
//Local Anisotropic Fourier Shell Correlation//
///////////////////////////////////////////////

void d_LocalAnisotropicFSC(tfloat* d_volume1, tfloat* d_volume2, int3 dimsvolume, tfloat* d_resolution, int windowsize, int maxradius, int2 anglesteps, tfloat threshold)
{
	//maxradius *= 2;

	//tfloat* h_volume1 = (tfloat*)MallocPinnedFromDeviceArray(*d_volume1, Elements(dimsvolume) * sizeof(tfloat));
	//tfloat* h_volume2 = (tfloat*)MallocPinnedFromDeviceArray(*d_volume2, Elements(dimsvolume) * sizeof(tfloat));

	int samplesperelement = anglesteps.x * anglesteps.y;

	int3 dimspaddedvolume = toInt3(dimsvolume.x + windowsize, dimsvolume.y + windowsize, dimsvolume.z + windowsize);
	int3 dimswindow = toInt3(windowsize, windowsize, windowsize);
	int3 dimspaddedwindow = toInt3(windowsize * 2, windowsize * 2, windowsize * 2);

	tfloat* d_paddedvolume1;
	cudaMalloc((void**)&d_paddedvolume1, Elements(dimspaddedvolume) * sizeof(tfloat));
	d_Pad(d_volume1, d_paddedvolume1, dimsvolume, dimspaddedvolume, T_PAD_MIRROR, (tfloat)0);
	//cudaFree(*d_volume1);
	tfloat* d_paddedvolume2;
	cudaMalloc((void**)&d_paddedvolume2, Elements(dimspaddedvolume) * sizeof(tfloat));
	d_Pad(d_volume2, d_paddedvolume2, dimsvolume, dimspaddedvolume, T_PAD_MIRROR, (tfloat)0);
	//cudaFree(*d_volume2);

	uint batchmemory = 256 * 1024 * 1024;
	uint windowmemory = Elements(dimswindow) * sizeof(tfloat);
	uint batchsize = batchmemory / windowmemory;

	tfloat *d_extracts1, *d_extracts2;
	cudaMalloc((void**)&d_extracts1, Elements(dimswindow) * batchsize * sizeof(tfloat));
	cudaMalloc((void**)&d_extracts2, Elements(dimswindow) * batchsize * sizeof(tfloat));
	/*tfloat *d_paddedextracts1, *d_paddedextracts2;
	cudaMalloc((void**)&d_paddedextracts1, Elements(dimspaddedwindow) * batchsize * sizeof(tfloat));
	cudaMalloc((void**)&d_paddedextracts2, Elements(dimspaddedwindow) * batchsize * sizeof(tfloat));*/

	tfloat* d_mask = CudaMallocValueFilled(Elements(dimswindow) * batchsize, (tfloat)1);
	d_HannMask(d_mask, d_mask, dimswindow, NULL, NULL, batchsize);
	//tfloat* h_mask = (tfloat*)MallocFromDeviceArray(d_mask, Elements(dimswindow) * batchsize * sizeof(tfloat));
	//free(h_mask);

	int3* h_extractcenters;
	cudaMallocHost((void**)&h_extractcenters, batchsize * sizeof(int3));
	int3* d_extractcenters;
	cudaMalloc((void**)&d_extractcenters, batchsize * sizeof(int3));

	tfloat* d_fsccurves;
	cudaMalloc((void**)&d_fsccurves, maxradius * samplesperelement * batchsize * sizeof(tfloat));
	tfloat* d_resvalues;
	cudaMalloc((void**)&d_resvalues, batchsize * samplesperelement * sizeof(tfloat));

	cufftHandle planforw = d_FFTR2CGetPlan(DimensionCount(dimswindow), dimswindow, batchsize);

	int3 dimstrimmed = toInt3(dimsvolume.x - windowsize, dimsvolume.y - windowsize, dimsvolume.z - windowsize);

	int elements = Elements(dimsvolume);
	int elementsxy = dimsvolume.x * dimsvolume.y;
	int elementswindow = Elements(dimswindow);


	for (int i = 0; i < elements; i += batchsize)
	{
		for(int b = i; b < min(elements, i + batchsize); b++)
		{
			int z = b / elementsxy;
			int y = (b - z * elementsxy) / dimsvolume.x;
			int x = b % dimsvolume.x;

			h_extractcenters[b - i] = toInt3(x, y, z);
		}
		cudaMemcpy(d_extractcenters, h_extractcenters, batchsize * sizeof(int3), cudaMemcpyHostToDevice);

		d_ExtractMany(d_paddedvolume1, d_extracts1, dimspaddedvolume, dimswindow, d_extractcenters, batchsize);
		d_ExtractMany(d_paddedvolume2, d_extracts2, dimspaddedvolume, dimswindow, d_extractcenters, batchsize);

		d_MultiplyByVector(d_extracts1, d_mask, d_extracts1, elementswindow, batchsize);
		d_MultiplyByVector(d_extracts2, d_mask, d_extracts2, elementswindow, batchsize);
		
		d_AnisotropicFSCMap(d_extracts1, d_extracts2, dimswindow, d_fsccurves, anglesteps, maxradius, T_FSC_MODE::T_FSC_THRESHOLD, threshold, &planforw, batchsize);

		d_ValueFill(d_resvalues, batchsize * samplesperelement, (tfloat)-1);
		d_FirstIndexOf(d_fsccurves, d_resvalues, maxradius, threshold, T_INTERP_LINEAR, batchsize * samplesperelement);
		/*tfloat* h_resvalues = (tfloat*)MallocFromDeviceArray(d_resvalues, batchsize * samplesperelement * sizeof(tfloat));
		free(h_resvalues);*/

		cudaMemcpy(d_resolution + i * samplesperelement, d_resvalues, min(batchsize, elements - i) * samplesperelement * sizeof(tfloat), cudaMemcpyDeviceToDevice);

		//break;
		printf("%f\n", (tfloat)i / (tfloat)elements * (tfloat)100);
	}


	cufftDestroy(planforw);
	cudaFree(d_resvalues);
	cudaFreeHost(d_fsccurves);
	cudaFree(d_extractcenters);
	cudaFreeHost(h_extractcenters);
	cudaFree(d_mask);
	/*cudaFree(d_paddedextracts2);
	cudaFree(d_paddedextracts1);*/
	cudaFree(d_extracts2);
	cudaFree(d_extracts1);
	cudaFree(d_paddedvolume1);
	cudaFree(d_paddedvolume2);
}