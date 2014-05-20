
#include "Prerequisites.h"

TEST(MEXProfiling, TSAPreprocess)
{
	cudaDeviceReset();

	//Case 1:
	{
		int3 dimsinput = toInt3(256, 256, 11);
		int nimages = dimsinput.z;
		int3 dimsimage = toInt3(dimsinput.x, dimsinput.y, 1);
		int3 dimscropped = toInt3((int)((double)dimsimage.x * 0.4), (int)((double)dimsimage.y * 0.4), 1);

		tfloat2* h_translations = (tfloat2*)MallocValueFilled(nimages * 2, (tfloat)0);
		tfloat* h_rotations = MallocValueFilled(nimages, (tfloat)0);
		tfloat2* h_prevtranslations = (tfloat2*)MallocValueFilled(nimages * 2, (tfloat)0);
		tfloat* h_prevrotations = MallocValueFilled(nimages, (tfloat)0);

		tfloat* d_images = NULL;
		tfloat* d_imagescropped = NULL;
		tcomplex* d_imagesft = NULL;
		tcomplex* d_imagesfttransformed = NULL;

		bool isfirsttime = false;
		if(d_images == NULL)
		{
			isfirsttime = true;

			d_images = CudaMallocValueFilled(Elements(dimsinput), (tfloat)0);
			cudaMalloc((void**)&d_imagescropped, Elements(dimscropped) * nimages * sizeof(tfloat));
			cudaMalloc((void**)&d_imagesft, ElementsFFT(dimsinput) * sizeof(tcomplex));
			cudaMalloc((void**)&d_imagesfttransformed, ElementsFFT(dimsinput) * sizeof(tcomplex));
		}

		bool* h_needupdate = (bool*)malloc(nimages * sizeof(bool));
		for (int n = 0; n < nimages; n++)
			h_needupdate[n] = (h_translations[n].x != h_prevtranslations[n].x || h_translations[n].y != h_prevtranslations[n].y || h_rotations[n] != h_prevrotations[n] || isfirsttime);
	
		tfloat* d_croppedbuffer;
		cudaMalloc((void**)&d_croppedbuffer, ElementsFFT(dimsimage) * sizeof(tcomplex));
		cufftHandle planforw = d_FFTR2CGetPlan(2, dimsimage);
		cufftHandle planback = d_IFFTC2RGetPlan(2, dimsimage);

		tfloat* d_ramp;
		cudaMalloc((void**)&d_ramp, ElementsFFT(dimsimage) * sizeof(tfloat));
		tfloat* h_ramp = (tfloat*)malloc(ElementsFFT(dimsimage) * sizeof(tfloat));
		for (int j = 0; j < dimsimage.y; j++)
			for (int i = 0; i < dimsimage.x / 2 + 1; i++)
			{
				tfloat x = (tfloat)(dimsimage.x / 2 - i) / ((tfloat)dimsimage.x * (tfloat)0.5);
				tfloat y = (tfloat)(dimsimage.y / 2 - j) / ((tfloat)dimsimage.y * (tfloat)0.5);
				tfloat rad = x * x + y * y;
				h_ramp[j * (dimsimage.x / 2 + 1) + i] = exp(-rad / (tfloat)2);
			}
		cudaMemcpy(d_ramp, h_ramp, ElementsFFT(dimsimage) * sizeof(tfloat), cudaMemcpyHostToDevice);
		free(h_ramp);

		for (int n = 0; n < nimages; n++)
		{
			if(!h_needupdate[n])
				continue;

			tfloat* d_Nimages = d_images + Elements(dimsimage) * n;
			tfloat* d_Nimagescropped = d_imagescropped + Elements(dimscropped) * n;
			tcomplex* d_Nimagesft = d_imagesft + ElementsFFT(dimsimage) * n;
			tcomplex* d_Nimagesfttransformed = d_imagesfttransformed + ElementsFFT(dimsimage) * n;
		
			if(isfirsttime)
			{
				d_RemapFull2FullFFT(d_Nimages, (tfloat*)d_Nimagesft, dimsimage);
				d_FFTR2C((tfloat*)d_Nimagesft, d_Nimagesft, &planforw);
				d_RemapHalfFFT2Half(d_Nimagesft, d_Nimagesft, dimsimage);
			}

			d_Rotate2DFT(d_Nimagesft, d_Nimagesfttransformed, dimsimage, h_rotations[n], T_INTERP_CUBIC);
			d_ComplexMultiplyByVector(d_Nimagesfttransformed, d_ramp, d_Nimagesfttransformed, ElementsFFT(dimsimage));

			d_RemapHalf2HalfFFT(d_Nimagesfttransformed, d_Nimagesfttransformed, dimsimage);
			tfloat3 delta = tfloat3(h_translations[n].x, h_translations[n].y, (tfloat)0);
			d_Shift(d_Nimagesfttransformed, d_Nimagesfttransformed, dimsimage, &delta, false);
		
			d_IFFTC2R(d_Nimagesfttransformed, d_croppedbuffer, &planback);
			d_RemapFullFFT2Full(d_croppedbuffer, d_croppedbuffer, dimsimage);
			d_Pad(d_croppedbuffer, d_Nimagescropped, dimsimage, dimscropped, T_PAD_VALUE, (tfloat)0);
			d_Norm(d_Nimagescropped, d_Nimagescropped, Elements(dimscropped), (char*)NULL, T_NORM_MEAN01STD, 0);
		}

		cufftDestroy(planback);
		cufftDestroy(planforw);
		cudaFree(d_ramp);
		cudaFree(d_croppedbuffer);

		cudaFree(d_images);
		cudaFree(d_imagescropped);
		cudaFree(d_imagesft);
		cudaFree(d_imagesfttransformed);

		free(h_needupdate);
	}

	cudaDeviceReset();
}