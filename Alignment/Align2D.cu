#include "../Prerequisites.cuh"
#include "../Functions.cuh"


////////////////////////////////////
//Equivalent of tom_os3_alignStack//
////////////////////////////////////

void d_Align2D(tfloat* d_input, tfloat* d_targets, int3 dims, int numtargets, tfloat3* d_params, int* d_membership, tfloat* d_scores, int maxtranslation, tfloat maxrotation, int iterations, T_ALIGN_MODE mode, int batch)
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

	#pragma region Masks

	tfloat* d_maskcart = CudaMallocValueFilled(Elements(effdims), (tfloat)1 / (tfloat)Elements(effdims));
	tfloat* d_maskpolar;
	cudaMalloc((void**)&d_maskpolar, Elements(polardims) * sizeof(tfloat));
	{
		tfloat fmaxtranslation = (tfloat)(maxtranslation + 1);
		d_SphereMask(d_maskcart, d_maskcart, effdims, &fmaxtranslation, (tfloat)1, (tfloat3*)NULL);
		tfloat* h_maskcart = (tfloat*)MallocFromDeviceArray(d_maskcart, Elements(effdims) * sizeof(tfloat));
		free(h_maskcart);
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
	int* h_membership = (int*)malloc(batch * sizeof(int));
	tfloat* h_scores = MallocValueFilled(batch * numtargets, (tfloat)0);
	tfloat3* h_intermedparams = (tfloat3*)malloc(batch * numtargets * sizeof(tfloat3));
	for (int t = 0; t < numtargets; t++)
		cudaMemcpy(h_intermedparams + t * batch, d_params, batch * sizeof(tfloat3), cudaMemcpyDeviceToHost);
	tfloat3* h_peakpos = (tfloat3*)malloc(batch * sizeof(tfloat3));
	tfloat* h_peakvalues = (tfloat*)malloc(batch * sizeof(tfloat));

	tfloat2* h_scale = (tfloat2*)MallocValueFilled(batch * 2, (tfloat)1);
	tfloat* h_rotation = (tfloat*)malloc(batch * sizeof(tfloat));
	tfloat2* h_translation = (tfloat2*)malloc(batch * sizeof(tfloat2));
	tfloat2* d_translation = (tfloat2*)CudaMallocValueFilled(batch * 2, (tfloat)0);

	for (int iteration = 0; iteration < iterations; iteration++)
	{
		if(mode & T_ALIGN_MODE::T_ALIGN_ROT)
		{
			for (int t = 0; t < numtargets; t++)
			{
				memcpy(h_params, h_intermedparams + batch * t, batch * sizeof(tfloat3));
				for (int b = 0; b < batch; b++)
					h_translation[b] = tfloat2((tfloat)(h_atlascoords[b].x + padding) + h_params[b].x, (tfloat)(h_atlascoords[b].y + padding) + h_params[b].y);
				cudaMemcpy(d_translation, h_translation, batch * sizeof(tfloat2), cudaMemcpyHostToDevice);

				d_CartAtlas2Polar(d_atlas, d_datapolar, d_translation, toInt2(atlasdims.x, atlasdims.y), toInt2(effdims.x, effdims.y), T_INTERP_LINEAR, batch);
				tfloat* h_datapolar = (tfloat*)MallocFromDeviceArray(d_datapolar, Elements(polardims) * batch * sizeof(tfloat));
				free(h_datapolar);

				d_NormMonolithic(d_datapolar, d_datapolar, Elements(polardims), T_NORM_MEAN01STD, batch);
				d_FFTR2C(d_datapolar, d_datapolarFFT, 2, polardims, batch);
				d_ComplexMultiplyByConjVector(d_datapolarFFT, d_targetspolarFFT + ElementsFFT(polardims) * t, d_datapolarFFT, ElementsFFT(polardims), batch);
				d_IFFTC2R(d_datapolarFFT, d_datapolar, 2, polardims, batch);
				d_Extract(d_datapolar, d_polarextract, polardims, toInt3(1, polardims.y, 1), toInt3(0, polardims.y / 2, 0), batch);

				d_Peak(d_polarextract, d_peakpos, d_peakvalues, toInt3(polardims.y, 1, 1), T_PEAK_SUBCOARSE, batch);

				cudaMemcpy(h_peakpos, d_peakpos, batch * sizeof(tfloat3), cudaMemcpyDeviceToHost);
				cudaMemcpy(h_peakvalues, d_peakvalues, batch * sizeof(tfloat), cudaMemcpyDeviceToHost);

				for (int b = 0; b < batch; b++)
				{
					if(h_peakvalues[b] > h_scores[batch * t + b])
					{
						h_intermedparams[batch * t + b].z = h_peakpos[b].x / (tfloat)polardims.y * PI2;
						h_scores[batch * t + b] = h_peakvalues[b];
					}
				}
			}
		}

		if(mode & T_ALIGN_MODE::T_ALIGN_TRANS)
		{
			for (int t = 0; t < numtargets; t++)
			{
				memcpy(h_params, h_intermedparams + batch * t, batch * sizeof(tfloat3));
				for (int b = 0; b < batch; b++)
				{
					h_rotation[b] = h_params[b].z;
					h_translation[b] = tfloat2((tfloat)h_atlascoords[b].x + (tfloat)(dims.x / 2), (tfloat)h_atlascoords[b].y + (tfloat)(dims.y / 2));
				}

				d_Extract2DTransformed(d_atlas, d_datacart, atlasdims, effdims, h_scale, h_rotation, h_translation, T_INTERP_LINEAR, batch);
				d_NormMonolithic(d_datacart, d_datacart, Elements(effdims), T_NORM_MEAN01STD, batch);
				d_FFTR2C(d_datacart, d_datacartFFT, 2, effdims, batch);
				d_ComplexMultiplyByConjVector(d_datacartFFT, d_targetscartFFT + ElementsFFT(effdims) * t, d_datacartFFT, ElementsFFT(effdims), batch);
				d_IFFTC2R(d_datacartFFT, d_datacart, 2, effdims, batch);
				d_RemapFullFFT2Full(d_datacart, d_datacart, effdims, batch);
				d_MultiplyByVector(d_datacart, d_maskcart, d_datacart, Elements(effdims), batch);

				d_Peak(d_datacart, d_peakpos, d_peakvalues, effdims, T_PEAK_SUBCOARSE, batch);
				d_SubtractScalar((tfloat*)d_peakpos, (tfloat*)d_peakpos, batch * 3, (tfloat)(effdims.x / 2));

				cudaMemcpy(h_peakpos, d_peakpos, batch * sizeof(tfloat3), cudaMemcpyDeviceToHost);
				cudaMemcpy(h_peakvalues, d_peakvalues, batch * sizeof(tfloat), cudaMemcpyDeviceToHost);

				for (int b = 0; b < batch; b++)
				{
					if(h_peakvalues[b] > h_scores[batch * t + b])
					{
						h_intermedparams[batch * t + b].x = h_peakpos[b].x;
						h_intermedparams[batch * t + b].y = h_peakpos[b].y;
						h_scores[batch * t + b] = h_peakvalues[b];
					}
				}
			}
		}

		if(mode != T_ALIGN_BOTH)
			break;
	}

	#pragma region AssignMembership

	for (int b = 0; b < batch; b++)
	{
		tfloat bestscore = (tfloat)-999;
		for (int t = 0; t < numtargets; t++)
		{
			if(h_scores[batch * t + b] > bestscore)
			{
				bestscore = h_scores[batch * t + b];
				h_params[b] = h_intermedparams[batch * t + b];
				h_membership[b] = t;
			}
		}
	}

	#pragma endregion

	#pragma region Cleanup

	free(h_translation);
	free(h_rotation);
	free(h_scale);
	free(h_peakvalues);
	free(h_peakpos);
	free(h_intermedparams);
	free(h_scores);
	free(h_membership);
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