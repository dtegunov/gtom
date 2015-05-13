#include "Prerequisites.cuh"
#include "FFT.cuh"
#include "Helper.cuh"
#include "Generics.cuh"
#include "Masking.cuh"
#include "Resolution.cuh"


///////////////////////////////////////////////
//Local Anisotropic Fourier Shell Correlation//
///////////////////////////////////////////////

void d_LocalAnisotropicFSC(tfloat* d_volume1, tfloat* d_volume2, int3 dimsvolume, uint nvolumes, vector<float3> v_directions, float coneangle, tfloat* d_resolution, int windowsize, int maxradius, tfloat threshold)
{
	// Fill output volume with 1, as regions closer than windowsize/2 to border can't be assessed
	d_ValueFill(d_resolution, Elements(dimsvolume), (tfloat)1);
	// dimsvolume sans the region where window around position of interest would exceed the volume
	int3 dimsaccessiblevolume = toInt3(dimsvolume.x - windowsize, dimsvolume.y - windowsize, dimsvolume.z - windowsize);
	int3 dimswindow = toInt3(windowsize, windowsize, windowsize);

	uint batchmemory = 128 * 1024 * 1024;
	uint windowmemory = Elements(dimswindow) * sizeof(tfloat);
	uint batchsize = batchmemory / windowmemory;

	// Allocate buffers for batch window extraction
	tfloat *d_extracts1, *d_extracts2;
	cudaMalloc((void**)&d_extracts1, Elements(dimswindow) * batchsize * sizeof(tfloat));
	cudaMalloc((void**)&d_extracts2, Elements(dimswindow) * batchsize * sizeof(tfloat));

	// ... and their FT
	tcomplex* d_extractsft1, *d_extractsft2;
	cudaMalloc((void**)&d_extractsft1, ElementsFFT(dimswindow) * batchsize * sizeof(tcomplex));
	cudaMalloc((void**)&d_extractsft2, ElementsFFT(dimswindow) * batchsize * sizeof(tcomplex));

	// Hann mask for extracted portions
	tfloat* d_mask = CudaMallocValueFilled(Elements(dimswindow), (tfloat)1);
	d_HannMask(d_mask, d_mask, dimswindow, NULL, NULL);
	//d_WriteMRC(d_mask, dimswindow, "d_mask.mrc");

	// Positions at which the windows will be extracted
	int3* h_extractorigins;
	cudaMallocHost((void**)&h_extractorigins, batchsize * sizeof(int3));
	int3* d_extractorigins;
	cudaMalloc((void**)&d_extractorigins, batchsize * sizeof(int3));

	// Addresses used to remap resolution values within smaller accessible volume to the larger overall volume
	intptr_t* h_remapaddresses;
	cudaMallocHost((void**)&h_remapaddresses, batchsize * sizeof(intptr_t));
	intptr_t* d_remapaddresses;
	cudaMalloc((void**)&d_remapaddresses, batchsize * sizeof(intptr_t));

	// FSC precursor data
	tfloat* d_fscnums, *d_fscdenoms1, *d_fscdenoms2;
	cudaMalloc((void**)&d_fscnums, nvolumes * v_directions.size() * maxradius * batchsize * sizeof(tfloat));
	cudaMalloc((void**)&d_fscdenoms1, nvolumes * v_directions.size() * maxradius * batchsize * sizeof(tfloat));
	cudaMalloc((void**)&d_fscdenoms2, nvolumes * v_directions.size() * maxradius * batchsize * sizeof(tfloat));

	// Buffers for calculated FSC curves and the resolution values derived from them
	tfloat* d_fsccurves;
	cudaMalloc((void**)&d_fsccurves, v_directions.size() * maxradius * batchsize * sizeof(tfloat));
	tfloat* d_resvalues;
	cudaMalloc((void**)&d_resvalues, v_directions.size() * maxradius * batchsize * sizeof(tfloat));

	// Batch FFT for extracted windows
	cufftHandle planforw = d_FFTR2CGetPlan(3, dimswindow, batchsize);

	int elementsvol = Elements(dimsaccessiblevolume);
	int elementsslice = dimsaccessiblevolume.x * dimsaccessiblevolume.y;
	int elementswindow = Elements(dimswindow);

	for (int i = 0; i < elementsvol; i += batchsize)
	{
		uint curbatch = min(batchsize, elementsvol - i);

		for (int b = 0; b < curbatch; b++)
		{
			// Set origins for window extraction
			int z = (i + b) / elementsslice;
			int y = ((i + b) % elementsslice) / dimsaccessiblevolume.x;
			int x = (i + b) % dimsaccessiblevolume.x;
			h_extractorigins[b] = toInt3(x, y, z);

			// Set remap addresses to get resolution values back into the larger overall volume
			x += windowsize / 2;
			y += windowsize / 2;
			z += windowsize / 2;
			h_remapaddresses[b] = (z * dimsvolume.y + y) * dimsvolume.x + x;
		}
		cudaMemcpy(d_extractorigins, h_extractorigins, curbatch * sizeof(int3), cudaMemcpyHostToDevice);
		cudaMemcpy(d_remapaddresses, h_remapaddresses, curbatch * sizeof(intptr_t), cudaMemcpyHostToDevice);

		for (uint v = 0; v < nvolumes; v++)
		{
			// Extract windows
			d_ExtractMany(d_volume1, d_extracts1, dimsvolume, dimswindow, d_extractorigins, curbatch);
			d_ExtractMany(d_volume2, d_extracts2, dimsvolume, dimswindow, d_extractorigins, curbatch);

			// Multiply by Hann mask
			d_MultiplyByVector(d_extracts1, d_mask, d_extracts1, elementswindow, curbatch);
			d_MultiplyByVector(d_extracts2, d_mask, d_extracts2, elementswindow, curbatch);
			//d_WriteMRC(d_extracts1, toInt3(dimswindow.x, dimswindow.y, dimswindow.z * 10), "d_extracts1.mrc");

			// FFT
			d_FFTR2C(d_extracts1, d_extractsft1, &planforw);
			d_FFTR2C(d_extracts2, d_extractsft2, &planforw);
			
			// Calculate FSC precursor data for this volume
			for (uint a = 0; a < v_directions.size(); a++)
				d_AnisotropicFSC(d_extractsft1,
								 d_extractsft2,
								 dimswindow,
								 d_fsccurves,
								 maxradius,
								 v_directions[a],
								 coneangle,
								 d_fscnums + (a * nvolumes + v) * maxradius * batchsize,
								 d_fscdenoms1 + (a * nvolumes + v) * maxradius * batchsize,
								 d_fscdenoms2 + (a * nvolumes + v) * maxradius * batchsize,
								 curbatch);
		}

		// Sum up FSC precursor data
		d_ReduceAdd(d_fscnums, d_fscnums, maxradius * batchsize, nvolumes, v_directions.size());
		d_ReduceAdd(d_fscdenoms1, d_fscdenoms1, maxradius * batchsize, nvolumes, v_directions.size());
		d_ReduceAdd(d_fscdenoms2, d_fscdenoms2, maxradius * batchsize, nvolumes, v_directions.size());

		// Calculate FSC curves
		d_MultiplyByVector(d_fscdenoms1, d_fscdenoms2, d_fscdenoms1, maxradius * curbatch * v_directions.size());
		d_Sqrt(d_fscdenoms1, d_fscdenoms1, maxradius * curbatch * v_directions.size());
		d_DivideByVector(d_fscnums, d_fscdenoms1, d_resvalues, maxradius * curbatch * v_directions.size());

		//d_WriteMRC(d_resvalues, toInt3(maxradius, curbatch, 1), "d_resvalues.mrc");

		// Get FSC value at threshold
		d_ValueFill(d_resvalues, curbatch * v_directions.size(), windowsize / (tfloat)2);
		d_FirstIndexOf(d_fsccurves, d_resvalues, maxradius, threshold, T_INTERP_LINEAR, curbatch * v_directions.size());

		// Remap back into the larger overall volume
		for (uint a = 0; a < v_directions.size(); a++)
			d_RemapReverse(d_resvalues + a * curbatch, d_remapaddresses, d_resolution + Elements(dimsvolume) * a, curbatch, Elements(dimsvolume), (tfloat)0);

		printf("%f\n", (tfloat)i / (tfloat)elementsvol * (tfloat)100);
	}


	cufftDestroy(planforw);

	cudaFree(d_resvalues);
	cudaFree(d_fscdenoms2);
	cudaFree(d_fscdenoms1);
	cudaFree(d_fscnums);
	cudaFree(d_fsccurves);
	cudaFree(d_remapaddresses);
	cudaFree(d_extractorigins);
	cudaFree(d_mask);
	cudaFree(d_extractsft2);
	cudaFree(d_extractsft1);
	cudaFree(d_extracts2);
	cudaFree(d_extracts1);

	cudaFreeHost(h_extractorigins);
	cudaFreeHost(h_remapaddresses);
}