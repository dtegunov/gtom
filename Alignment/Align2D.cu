#include "../Prerequisites.cuh"
#include "../Functions.cuh"


////////////////////////////////////
//Equivalent of tom_os3_alignStack//
////////////////////////////////////

void d_Align2D(tfloat* d_input, tfloat* d_targets, int3 dims, int numtargets, tfloat3* d_params, tfloat* d_scores, int maxtranslation, tfloat maxrotation, int iterations, T_ALIGN_MODE mode, int batch)
{
	int padding = max(dims.x / 2 - (int)((tfloat)1 / (sin(min(ToRad(90), ToRad(45) + maxrotation)) / sin(ToRad(45))) * (tfloat)(dims.x / 2)), maxtranslation);
	int3 effdims = toInt3(dims.x - padding * 2, dims.y - padding * 2, 1);
	int3 polardims = toInt3(GetCart2PolarSize(toInt2(effdims.x, effdims.y)));
	
	#pragma region Targets

	tcomplex* d_targetscartFFT;
	cudaMalloc((void**)&d_targetscartFFT, ElementsFFT(effdims) * numtargets * sizeof(tcomplex));
	tcomplex* d_targetspolarFFT;
	cudaMalloc((void**)&d_targetspolarFFT, ElementsFFT(polardims) * numtargets * sizeof(tcomplex));
	{
		tfloat* d_targetscart;
		cudaMalloc((void**)&d_targetscart, Elements(effdims) * numtargets * sizeof(tfloat));
		tfloat* d_targetspolar;
		cudaMalloc((void**)&d_targetspolar, Elements(polardims) * numtargets * sizeof(tfloat));

		d_Extract(d_targets, d_targetscart, dims, effdims, toInt3(dims.x / 2, dims.y / 2, 0), numtargets);
		d_Cart2Polar(d_targetscart, d_targetspolar, toInt2(effdims.x, effdims.y), T_INTERP_CUBIC, numtargets);

		d_NormMonolithic(d_targetscart, d_targetscart, Elements(effdims), T_NORM_MEAN01STD, numtargets);
		d_NormMonolithic(d_targetspolar, d_targetspolar, Elements(polardims), T_NORM_MEAN01STD, numtargets);

		d_FFTR2C(d_targetscart, d_targetscartFFT, 2, effdims, numtargets);
		d_FFTR2C(d_targetspolar, d_targetspolarFFT, 2, polardims, numtargets);

		cudaFree(d_targetspolar);
		cudaFree(d_targetscart);
	}

	#pragma endregion

	#pragma region Atlas

	int sidelength = NextPow2((size_t)ceil(sqrt((tfloat)batch)) * (size_t)dims.x);
	int3 atlasdims = toInt3(sidelength, sidelength, 1);
	int atlasrow = atlasdims.x / dims.x;

	tfloat* d_atlas = CudaMallocValueFilled(Elements(atlasdims), (tfloat)0);
	int2* h_atlascoords = (int2*)malloc(batch * sizeof(int2));

	for (int b = 0; b < batch; b++)
	{
		int offsetx = (b % atlasrow) * dims.x;
		int offsety = (b / atlasrow) * dims.y;
		h_atlascoords[b] = toInt2(offsetx, offsety);
		for (int y = 0; y < dims.y; y++)
			cudaMemcpy(d_atlas + (offsety + y) * atlasdims.x + offsetx, d_input + b * Elements(dims) + y * dims.x, dims.x * sizeof(tfloat), cudaMemcpyDeviceToDevice);
	}

	#pragma endregion

	tfloat* d_datacart;
	cudaMalloc((void**)&d_datacart, Elements(effdims) * batch * sizeof(tfloat));
	tfloat* d_datapolar;
	cudaMalloc((void**)&d_datapolar, Elements(polardims) * batch * sizeof(tfloat));
	tcomplex* d_datacartFFT;
	cudaMalloc((void**)&d_datacartFFT, ElementsFFT(effdims) * batch * sizeof(tcomplex));
	tcomplex* d_datapolarFFT;
	cudaMalloc((void**)&d_datapolarFFT, ElementsFFT(polardims) * batch * sizeof(tcomplex));
	tfloat* d_polarextract;
	cudaMalloc((void**)&d_polarextract, polardims.y * batch * sizeof(tfloat));
	tfloat3* d_peakpos;
	cudaMalloc((void**)&d_peakpos, batch * sizeof(tfloat3));
	tfloat* d_peakvalues;
	cudaMalloc((void**)&d_peakvalues, batch * sizeof(tfloat));

	tfloat3* h_params = (tfloat3*)malloc(batch * sizeof(tfloat3));
	tfloat* h_scores = MallocValueFilled(batch, (tfloat)0);
	tfloat3* h_intermedparams = (tfloat3*)malloc(batch * numtargets * sizeof(tfloat3));
	for (int t = 0; t < numtargets; t++)
		cudaMemcpy(h_intermedparams + t * batch, d_params, batch * sizeof(tfloat3), cudaMemcpyDeviceToHost);

	tfloat2* h_scale = (tfloat2*)MallocValueFilled(batch * 2, (tfloat)1);
	tfloat* h_rotation = (tfloat*)malloc(batch * sizeof(tfloat));
	tfloat2* h_translation = (tfloat2*)malloc(batch * sizeof(tfloat2));

	for (int iteration = 0; iteration < iterations; iteration++)
	{
		if(mode & T_ALIGN_MODE::T_ALIGN_ROT)
		{
			
		}

		if(mode & T_ALIGN_MODE::T_ALIGN_TRANS)
		{
			for (int t = 0; t < numtargets; t++)
			{
				memcpy(h_params, h_intermedparams + batch * t, batch * sizeof(tfloat3));
				for (int b = 0; b < batch; b++)
				{
					h_rotation[b] = h_params[b].z;
					h_translation[b] = tfloat2((tfloat)h_atlascoords[b].x + h_params[b].x + (tfloat)(dims.x / 2), (tfloat)h_atlascoords[b].y + h_params[b].y + (tfloat)(dims.y / 2));
				}

				d_Extract2DTransformed(d_atlas, d_datacart, atlasdims, effdims, h_scale, h_rotation, h_translation, T_INTERP_LINEAR, batch);
				tcomplex* h_targetscartFFT = (tcomplex*)MallocFromDeviceArray(d_targetscartFFT, ElementsFFT(effdims) * sizeof(tcomplex));
				free(h_targetscartFFT);
				d_NormMonolithic(d_datacart, d_datacart, Elements(effdims), T_NORM_MEAN01STD, batch);
				d_FFTR2C(d_datacart, d_datacartFFT, 2, effdims, batch);
				d_ComplexMultiplyByConjVector(d_datacartFFT, d_targetscartFFT + ElementsFFT(effdims) * t, d_datacartFFT, ElementsFFT(effdims), batch);
				d_IFFTC2R(d_datacartFFT, d_datacart, 2, effdims, batch);
				d_RemapFullFFT2Full(d_datacart, d_datacart, effdims, batch);
				tfloat* h_datacart = (tfloat*)MallocFromDeviceArray(d_datacart, Elements(effdims) * sizeof(tfloat));
				free(h_datacart);

				d_Peak(d_datacart, d_peakpos, d_peakvalues, effdims, T_PEAK_SUBFINE, batch);

				tfloat3* h_peakpos = (tfloat3*)MallocFromDeviceArray(d_peakpos, batch * sizeof(tfloat3));
				free(h_peakpos);
			}
		}

		if(mode != T_ALIGN_BOTH)
			break;
	}

	#pragma region Cleanup

	free(h_translation);
	free(h_rotation);
	free(h_scale);
	free(h_intermedparams);
	free(h_scores);
	free(h_params);

	cudaFree(d_peakvalues);
	cudaFree(d_peakpos);
	cudaFree(d_polarextract);
	cudaFree(d_datapolarFFT);
	cudaFree(d_datacartFFT);
	cudaFree(d_datapolar);
	cudaFree(d_datacart);

	free(h_atlascoords);
	cudaFree(d_atlas);

	cudaFree(d_targetspolarFFT);
	cudaFree(d_targetscartFFT);

	#pragma endregion
}