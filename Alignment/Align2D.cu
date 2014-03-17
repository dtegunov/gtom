#include "../Prerequisites.cuh"
#include "../Functions.cuh"


////////////////////////////////////
//Equivalent of tom_os3_alignStack//
////////////////////////////////////

void d_Align2D(tfloat* d_input, tfloat* d_targets, int3 dims, int numtargets, tfloat3* h_params, int* h_membership, tfloat* h_scores, int maxtranslation, tfloat maxrotation, int iterations, T_ALIGN_MODE mode, int batch)
{
	int polarboost = 100;	//Sub-pixel precision for polar correlation peak
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

	int3 atlasdims = toInt3(1, 1, 1);
	int2 atlasprimitives = toInt2(1, 1);

	int2* h_atlascoords = (int2*)malloc(batch * sizeof(int2));
	tfloat* d_atlas = d_MakeAtlas(d_input, dims, atlasdims, atlasprimitives, h_atlascoords);
	int atlasrow = atlasprimitives.x;

	#pragma endregion

	#pragma region Masks

	tfloat* d_maskcart = CudaMallocValueFilled(Elements(effdims), (tfloat)1 / (tfloat)Elements(effdims));
	tfloat* d_maskpolar;
	cudaMalloc((void**)&d_maskpolar, polardims.y * polarboost * sizeof(tfloat));
	{
		tfloat fmaxtranslation = (tfloat)(maxtranslation + 1);
		d_SphereMask(d_maskcart, d_maskcart, effdims, &fmaxtranslation, (tfloat)1, (tfloat3*)NULL);
		
		tfloat* h_maskpolar = MallocValueFilled(polardims.y * polarboost, (tfloat)0);
		h_maskpolar[0] = (tfloat)1;
		for(int a = 1; a < (int)ceil(maxrotation / PI2 * (tfloat)(polardims.y * polarboost)); a++)
		{
			h_maskpolar[a] = (tfloat)1;
			h_maskpolar[polardims.y * polarboost - a] = (tfloat)1;
		}
		cudaMemcpy(d_maskpolar, h_maskpolar, polardims.y * polarboost * sizeof(tfloat), cudaMemcpyHostToDevice);
		free(h_maskpolar);
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
	tfloat* d_polarextractboost;
	cudaMalloc((void**)&d_polarextractboost, polardims.y * polarboost * batch * sizeof(tfloat));
	tfloat3* d_peakpos;
	cudaMalloc((void**)&d_peakpos, batch * sizeof(tfloat3));
	tfloat* d_peakvalues;
	cudaMalloc((void**)&d_peakvalues, batch * sizeof(tfloat));

	tfloat* h_scoresrot = MallocValueFilled(batch * numtargets, (tfloat)0);
	tfloat* h_scorestrans = MallocValueFilled(batch * numtargets, (tfloat)0);
	tfloat3* h_intermedparams = (tfloat3*)malloc(batch * numtargets * sizeof(tfloat3));
	for (int t = 0; t < numtargets; t++)
		memcpy(h_intermedparams + t * batch, h_params, batch * sizeof(tfloat3));
	tfloat3* h_peakpos = (tfloat3*)malloc(batch * sizeof(tfloat3));
	tfloat* h_peakvalues = (tfloat*)malloc(batch * sizeof(tfloat));

	tfloat2* h_scale = (tfloat2*)MallocValueFilled(batch * 2, (tfloat)1);
	tfloat* h_rotation = (tfloat*)malloc(batch * sizeof(tfloat));
	tfloat2* h_translation = (tfloat2*)malloc(batch * sizeof(tfloat2));
	tfloat2* d_translation = (tfloat2*)CudaMallocValueFilled(batch * 2, (tfloat)0);

	cufftHandle planforwTrans, planbackTrans;
	cufftHandle planforwRot, planbackRot;
	if(mode & T_ALIGN_MODE::T_ALIGN_ROT)
	{
		planforwRot = d_FFTR2CGetPlan(2, polardims, batch);
		planbackRot = d_IFFTC2RGetPlan(2, polardims, batch);
	}
	if(mode & T_ALIGN_MODE::T_ALIGN_ROT)
	{
		planforwTrans = d_FFTR2CGetPlan(2, effdims, batch);
		planbackTrans = d_IFFTC2RGetPlan(2, effdims, batch);
	}

	if(iterations == 0 && mode == T_ALIGN_BOTH)
	{
		for(double a = (double)-maxrotation; a < (double)maxrotation; a += (double)ToRad(0.5))
		{
			for (int t = 0; t < numtargets; t++)
			{
				memcpy(h_params, h_intermedparams + batch * t, batch * sizeof(tfloat3));
				for (int b = 0; b < batch; b++)
				{
					h_rotation[b] = (tfloat)a;
					h_translation[b] = tfloat2((tfloat)h_atlascoords[b].x + (tfloat)(dims.x / 2), (tfloat)h_atlascoords[b].y + (tfloat)(dims.y / 2));
				}

				d_Extract2DTransformed(d_atlas, d_datacart, atlasdims, effdims, h_scale, h_rotation, h_translation, T_INTERP_LINEAR, batch);
				d_NormMonolithic(d_datacart, d_datacart, Elements(effdims), T_NORM_MEAN01STD, batch);
				d_FFTR2C(d_datacart, d_datacartFFT, &planforwTrans);
				d_ComplexMultiplyByConjVector(d_datacartFFT, d_targetscartFFT + ElementsFFT(effdims) * t, d_datacartFFT, ElementsFFT(effdims), batch);
				d_IFFTC2R(d_datacartFFT, d_datacart, &planbackTrans, effdims);
				d_RemapFullFFT2Full(d_datacart, d_datacart, effdims, batch);
				d_MultiplyByVector(d_datacart, d_maskcart, d_datacart, Elements(effdims), batch);

				d_Peak(d_datacart, d_peakpos, d_peakvalues, effdims, T_PEAK_INTEGER, batch);
				d_SubtractScalar((tfloat*)d_peakpos, (tfloat*)d_peakpos, batch * 3, (tfloat)(effdims.x / 2));
				d_MultiplyByScalar(d_peakvalues, d_peakvalues, batch, (tfloat)1 / (tfloat)Elements(effdims));

				cudaMemcpy(h_peakpos, d_peakpos, batch * sizeof(tfloat3), cudaMemcpyDeviceToHost);
				cudaMemcpy(h_peakvalues, d_peakvalues, batch * sizeof(tfloat), cudaMemcpyDeviceToHost);

				for (int b = 0; b < batch; b++)
				{
					if(h_peakvalues[b] > h_scorestrans[batch * t + b])
					{
						h_intermedparams[batch * t + b].x = h_peakpos[b].x;
						h_intermedparams[batch * t + b].y = h_peakpos[b].y;
						h_intermedparams[batch * t + b].z = (tfloat)a;
						h_scorestrans[batch * t + b] = h_peakvalues[b];
					}
				}
			}
		}
	}
	//else
	if(iterations == 0)
		iterations = 1;
	{
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
					d_FFTR2C(d_datapolar, d_datapolarFFT, &planforwRot);
					d_ComplexMultiplyByConjVector(d_datapolarFFT, d_targetspolarFFT + ElementsFFT(polardims) * t, d_datapolarFFT, ElementsFFT(polardims), batch);
					d_IFFTC2R(d_datapolarFFT, d_datapolar, &planbackRot, polardims);
					d_Extract(d_datapolar, d_polarextract, polardims, toInt3(1, polardims.y, 1), toInt3(0, polardims.y / 2, 0), batch);
					d_Scale(d_polarextract, d_polarextractboost, toInt3(polardims.y, 1, 1), toInt3(polardims.y * polarboost, 1, 1), T_INTERP_FOURIER, NULL, NULL, batch);

					d_MultiplyByVector(d_polarextractboost, d_maskpolar, d_polarextractboost, polardims.y * polarboost, batch);
					d_Peak(d_polarextractboost, d_peakpos, d_peakvalues, toInt3(polardims.y * polarboost, 1, 1), T_PEAK_INTEGER, batch);
					d_MultiplyByScalar(d_peakvalues, d_peakvalues, batch, (tfloat)1 / (tfloat)Elements(polardims));

					cudaMemcpy(h_peakpos, d_peakpos, batch * sizeof(tfloat3), cudaMemcpyDeviceToHost);
					cudaMemcpy(h_peakvalues, d_peakvalues, batch * sizeof(tfloat), cudaMemcpyDeviceToHost);

					for (int b = 0; b < batch; b++)
					{
						h_peakvalues[b] /= (tfloat)Elements(polardims);
						if(h_peakvalues[b] > h_scoresrot[batch * t + b])
						{
							if(abs(h_peakpos[b].x - (tfloat)(polardims.y * polarboost)) < h_peakpos[b].x)
								h_peakpos[b].x = h_peakpos[b].x - (tfloat)(polardims.y * polarboost);

							h_intermedparams[batch * t + b].z = h_peakpos[b].x / (tfloat)(polardims.y * polarboost) * PI2;
							h_scoresrot[batch * t + b] = h_peakvalues[b];
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
					d_FFTR2C(d_datacart, d_datacartFFT, &planforwTrans);
					d_ComplexMultiplyByConjVector(d_datacartFFT, d_targetscartFFT + ElementsFFT(effdims) * t, d_datacartFFT, ElementsFFT(effdims), batch);
					d_IFFTC2R(d_datacartFFT, d_datacart, &planbackTrans, effdims);
					d_RemapFullFFT2Full(d_datacart, d_datacart, effdims, batch);
					d_MultiplyByVector(d_datacart, d_maskcart, d_datacart, Elements(effdims), batch);

					d_Peak(d_datacart, d_peakpos, d_peakvalues, effdims, T_PEAK_SUBCOARSE, batch);
					d_SubtractScalar((tfloat*)d_peakpos, (tfloat*)d_peakpos, batch * 3, (tfloat)(effdims.x / 2));
					d_MultiplyByScalar(d_peakvalues, d_peakvalues, batch, (tfloat)1 / (tfloat)Elements(effdims));

					cudaMemcpy(h_peakpos, d_peakpos, batch * sizeof(tfloat3), cudaMemcpyDeviceToHost);
					cudaMemcpy(h_peakvalues, d_peakvalues, batch * sizeof(tfloat), cudaMemcpyDeviceToHost);

					for (int b = 0; b < batch; b++)
					{
						if(h_peakvalues[b] > h_scorestrans[batch * t + b])
						{
							h_intermedparams[batch * t + b].x = h_peakpos[b].x;
							h_intermedparams[batch * t + b].y = h_peakpos[b].y;
							h_scorestrans[batch * t + b] = h_peakvalues[b];
						}
					}
				}
			}

			if(mode != T_ALIGN_BOTH)
				break;
		}
	}

	#pragma region AssignMembership

	for (int b = 0; b < batch; b++)
	{
		tfloat bestscore = (tfloat)-999;
		for (int t = 0; t < numtargets; t++)
		{
			if(max(h_scoresrot[batch * t + b], h_scorestrans[batch * t + b]) > bestscore)
			{
				bestscore = max(h_scoresrot[batch * t + b], h_scorestrans[batch * t + b]);
				h_params[b] = h_intermedparams[batch * t + b];
				h_membership[b] = t;
			}
		}
		h_scores[b] = bestscore;
	}

	#pragma endregion

	#pragma region Cleanup

	free(h_translation);
	free(h_rotation);
	free(h_scale);
	free(h_peakvalues);
	free(h_peakpos);
	free(h_intermedparams);
	free(h_scorestrans);
	free(h_scoresrot);
	
	if(mode & T_ALIGN_MODE::T_ALIGN_ROT)
	{
		cufftDestroy(planforwRot);
		cufftDestroy(planbackRot);
	}
	if(mode & T_ALIGN_MODE::T_ALIGN_ROT)
	{
		cufftDestroy(planforwTrans);
		cufftDestroy(planbackTrans);
	}

	cudaFree(d_peakvalues);
	cudaFree(d_peakpos);
	cudaFree(d_polarextractboost);
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