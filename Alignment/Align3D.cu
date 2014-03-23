#include "../Prerequisites.cuh"
#include "../Functions.cuh"
#include "../GLMFunctions.cuh"


/////////////////////////////////////////////////
//Aligns a 3D volume to one or multiple targets//
/////////////////////////////////////////////////

void d_Align3D(tfloat* d_input, 
			   tfloat* d_targets, 
			   int3 dims, 
			   int numtargets, 
			   tfloat3 &position, 
			   tfloat3 &rotation, 
			   int* h_membership, 
			   tfloat* h_scores, 
			   tfloat3* h_allpositions, 
			   tfloat3* h_allrotations, 
			   int maxtranslation, 
			   tfloat3 maxrotation, 
			   tfloat rotationstep, 
			   int rotationrefinements, 
			   T_ALIGN_MODE mode)
{
	if(mode == T_ALIGN_BOTH || mode == T_ALIGN_TRANS)
	{
		tcomplex* d_targetft;
		cudaMalloc((void**)&d_targetft, ElementsFFT(dims) * sizeof(tcomplex));
		tcomplex* d_rotatedft;
		cudaMalloc((void**)&d_rotatedft, ElementsFFT(dims) * sizeof(tcomplex));

		cudaFree(d_rotatedft);
		cudaFree(d_targetft);
	}
	else if(mode == T_ALIGN_ROT)
	{

		tfloat* d_normtargets;
		cudaMalloc((void**)&d_normtargets, Elements(dims) * numtargets * sizeof(tfloat));
		//d_Norm(d_targets, d_normtargets, Elements(dims), (char*)NULL, T_NORM_MEAN01STD, 0, numtargets);

		tfloat* d_translated;
		cudaMalloc((void**)&d_translated, Elements(dims) * sizeof(tfloat));
		if(position.x != 0 || position.y != 0 || position.z != 0)
			d_Shift(d_input, d_translated, dims, &position);
		else
			cudaMemcpy(d_translated, d_input, Elements(dims) * sizeof(tfloat), cudaMemcpyDeviceToDevice);

		tfloat* d_scaled;
		cudaMalloc((void**)&d_scaled, Elements(dims) * sizeof(tfloat));

		for (int t = 0; t < numtargets; t++)
		{
			h_scores[t] = (tfloat)-1;
			h_allpositions[t] = position;
			h_allrotations[t] = rotation;
		}		
		
		{
			int pixelsperstep = max((int)floor(sqrt(pow(sin(rotationstep) * 0.25f * (tfloat)dims.x, 2.0f) + pow((1.0f - cos(rotationstep)) * 0.25f * (tfloat)dims.x, 2.0f))), 1);
			int newdim = max(NextMultipleOf(dims.x / pixelsperstep / 2 * 2, 4), 8);
			int3 dimsscaled = toInt3(newdim, newdim, newdim);

			long memory = 1 * 1024 * 1024 * 1024;
			long memperprimitive = Elements(dimsscaled) * 2 * sizeof(tfloat);
			int batchsize = min(memory / memperprimitive, 65535);

			tfloat* d_rotated;
			cudaMalloc((void**)&d_rotated, Elements(dimsscaled) * batchsize * sizeof(tfloat));
			tfloat* d_corr;
			cudaMalloc((void**)&d_corr, Elements(dimsscaled) * batchsize * sizeof(tfloat));
			tfloat* d_sum;
			cudaMalloc((void**)&d_sum, batchsize * sizeof(tfloat));
			tfloat* h_sum = (tfloat*)malloc(batchsize * sizeof(tfloat));

			if(pixelsperstep > 1)
			{
				d_Scale(d_translated, d_scaled, dims, dimsscaled, T_INTERP_FOURIER);
				d_Scale(d_targets, d_normtargets, dims, dimsscaled, T_INTERP_FOURIER, NULL, NULL, numtargets);
				d_Norm(d_normtargets, d_normtargets, Elements(dimsscaled), (char*)NULL, T_NORM_MEAN01STD, 0, numtargets);
			}
			else
			{
				cudaMemcpy(d_scaled, d_translated, Elements(dims) * sizeof(tfloat), cudaMemcpyDeviceToDevice);
				d_Norm(d_targets, d_normtargets, Elements(dims), (char*)NULL, T_NORM_MEAN01STD, 0, numtargets);
			}

			tfloat2 rotrange = tfloat2(rotation.x - maxrotation.x, rotation.x + maxrotation.x);
			tfloat2 tiltrange = tfloat2(max(rotation.y - maxrotation.y, (tfloat)0), min(rotation.y + maxrotation.y, PI));
			tfloat2 psirange = tfloat2(rotation.z - maxrotation.z, rotation.z + maxrotation.z);
			int numangles = 0;
			tfloat3* h_angles = GetEqualAngularSpacing(rotrange, tiltrange, psirange, rotationstep, numangles);

			for (int a = 0; a < numangles; a += batchsize)
			{
				int currentbatch = min(batchsize, numangles - a - 1);

				d_Rotate3D(d_scaled, d_rotated, dimsscaled, h_angles + a, T_INTERP_LINEAR, currentbatch);

				if(batchsize < 20)
					d_Norm(d_rotated, d_rotated, Elements(dimsscaled), (char*)NULL, T_NORM_MEAN01STD, 0, currentbatch);
				else
					d_NormMonolithic(d_rotated, d_rotated, Elements(dimsscaled), T_NORM_MEAN01STD, currentbatch);

				for(int t = 0; t < numtargets; t++)
				{
					d_MultiplyByVector(d_rotated, d_normtargets + t * Elements(dimsscaled), d_corr, Elements(dimsscaled), currentbatch);
					if(batchsize < 20)
						d_Sum(d_corr, d_sum, Elements(dimsscaled), currentbatch);
					else
						d_SumMonolithic(d_corr, d_sum, Elements(dimsscaled), currentbatch);
					cudaMemcpy(h_sum, d_sum, currentbatch * sizeof(tfloat), cudaMemcpyDeviceToHost);

					for (int b = 0; b < currentbatch; b++)
					{
						h_sum[b] /= (tfloat)Elements(dimsscaled);

						if(h_sum[b] > h_scores[t])
						{
							h_scores[t] = h_sum[b];
							h_allrotations[t] = h_angles[a + b];
						}
					}
				}
			}
			free(h_angles);
			free(h_sum);
			cudaFree(d_sum);
			cudaFree(d_corr);
			cudaFree(d_rotated);
		}
		rotationstep /= 5.0f;

		for (int i = 0; i < rotationrefinements; i++)
		{
			int pixelsperstep = max((int)floor(sqrt(pow(sin(rotationstep) * 0.25f * (tfloat)dims.x, 2.0f) + pow((1.0f - cos(rotationstep)) * 0.25f * (tfloat)dims.x, 2.0f))), 1);
			int newdim = max(NextMultipleOf(dims.x / pixelsperstep / 2 * 2, 4), 8);
			int3 dimsscaled = toInt3(newdim, newdim, newdim);

			long memory = 1 * 1024 * 1024 * 1024;
			long memperprimitive = Elements(dimsscaled) * 2 * sizeof(tfloat);
			int batchsize = min(memory / memperprimitive, 65535);

			tfloat* d_rotated;
			cudaMalloc((void**)&d_rotated, Elements(dimsscaled) * batchsize * sizeof(tfloat));
			tfloat* d_corr;
			cudaMalloc((void**)&d_corr, Elements(dimsscaled) * batchsize * sizeof(tfloat));
			tfloat* d_sum;
			cudaMalloc((void**)&d_sum, batchsize * sizeof(tfloat));
			tfloat* h_sum = (tfloat*)malloc(batchsize * sizeof(tfloat));

			if(pixelsperstep > 1)
			{
				d_Scale(d_translated, d_scaled, dims, dimsscaled, T_INTERP_FOURIER);
				d_Scale(d_targets, d_normtargets, dims, dimsscaled, T_INTERP_FOURIER, NULL, NULL, numtargets);
				d_Norm(d_normtargets, d_normtargets, Elements(dimsscaled), (char*)NULL, T_NORM_MEAN01STD, 0, numtargets);
			}
			else
			{
				cudaMemcpy(d_scaled, d_translated, Elements(dims) * sizeof(tfloat), cudaMemcpyDeviceToDevice);
				d_Norm(d_targets, d_normtargets, Elements(dims), (char*)NULL, T_NORM_MEAN01STD, 0, numtargets);
			}

			for(int t = 0; t < numtargets; t++)
			{
				tfloat2 rotrange = tfloat2(h_allrotations[t].x - rotationstep * 5.0f, h_allrotations[t].x + rotationstep * 5.0f);
				tfloat2 tiltrange = tfloat2(max(h_allrotations[t].y - rotationstep * 5.0f, (tfloat)0), min(h_allrotations[t].y + rotationstep * 5.0f, PI));
				tfloat2 psirange = tfloat2(h_allrotations[t].z - rotationstep * 5.0f, h_allrotations[t].z + rotationstep * 5.0f);
				int numangles = 0;
				tfloat3* h_angles = GetEqualAngularSpacing(rotrange, tiltrange, psirange, rotationstep, numangles);

				for (int a = 0; a < numangles; a += batchsize)
				{
					int currentbatch = min(batchsize, numangles - a - 1);

					d_Rotate3D(d_translated, d_rotated, dimsscaled, h_angles + a, T_INTERP_LINEAR, currentbatch);

					if(batchsize < 20)
						d_Norm(d_rotated, d_rotated, Elements(dimsscaled), (char*)NULL, T_NORM_MEAN01STD, 0, currentbatch);
					else
						d_NormMonolithic(d_rotated, d_rotated, Elements(dimsscaled), T_NORM_MEAN01STD, currentbatch);

						d_MultiplyByVector(d_rotated, d_normtargets + t * Elements(dimsscaled), d_corr, Elements(dimsscaled), currentbatch);
						if(batchsize < 20)
							d_Sum(d_corr, d_sum, Elements(dimsscaled), currentbatch);
						else
							d_SumMonolithic(d_corr, d_sum, Elements(dimsscaled), currentbatch);
						cudaMemcpy(h_sum, d_sum, currentbatch * sizeof(tfloat), cudaMemcpyDeviceToHost);

						for (int b = 0; b < currentbatch; b++)
						{
							h_sum[b] /= (tfloat)Elements(dimsscaled);

							if(h_sum[b] > h_scores[t])
							{
								h_scores[t] = h_sum[b];
								h_allrotations[t] = h_angles[a + b];
							}
						}
				}
				free(h_angles);
			}
			free(h_sum);
			cudaFree(d_sum);
			cudaFree(d_corr);
			cudaFree(d_rotated);

			rotationstep /= 5.0f;
		}

		cudaFree(d_scaled);
		cudaFree(d_translated);
		cudaFree(d_normtargets);
	}
}