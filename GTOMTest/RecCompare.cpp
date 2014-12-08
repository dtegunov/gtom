#include "Prerequisites.h"

TEST(Reconstruction, RecCompare)
{
	freopen("CONOUT$", "wb", stdout);

	cudaDeviceReset();

	//Case 1:
	{
		ofstream scores;
		scores.open(((string)("scores.txt")).c_str(), ios::out);

		int3 dimsvolume = toInt3(128, 64, 16);
		tfloat3 volumeoffset = tfloat3(0.0f, 0.0f, 0.0f);
		int2 dimsimage = toInt2(64, 64);
		int nimages = 61;
		tfloat3* h_angles = (tfloat3*)malloc(nimages * sizeof(tfloat3));
		tfloat2* h_translations = (tfloat2*)malloc(nimages * sizeof(tfloat2));
		tfloat2* h_scales = (tfloat2*)malloc(nimages * sizeof(tfloat2));
		for (int n = 0; n < nimages; n++)
		{
			h_angles[n] = tfloat3(0.0f, ToRad(-60.0f + n * 2.0f), 0.0f);
			h_translations[n] = tfloat2(0.0f, 0.0f);
			h_scales[n] = tfloat2(1.0f, 1.0f);
		}

		tfloat* d_volume = CudaMallocValueFilled(Elements(dimsvolume), 0.0f);
		tfloat* d_volume2 = CudaMallocValueFilled(64 * 64 * 64, 0.0f);
		tfloat* d_images = (tfloat*)CudaMallocFromBinaryFile("Data/Reconstruction/SIRTvsWBP.bin");
		tfloat* d_volumeoriginal = (tfloat*)CudaMallocFromBinaryFile("Data/Reconstruction/SIRTvsWBPvolume.bin");
		d_NormMonolithic(d_volumeoriginal, d_volumeoriginal, 64 * 64 * 16, T_NORM_MEAN01STD, 1);
		tfloat* d_reprojections = CudaMallocValueFilled(Elements2(dimsimage) * nimages, 0.0f);
		tfloat* d_masks = CudaMallocValueFilled(Elements2(dimsimage) * nimages, 1.0f);
		//d_SphereMask(d_masks, d_masks, toInt3(dimsimage), NULL, 0.0f, NULL, nimages);
		d_RectangleMask(d_masks, d_masks, toInt3(64, 64, 1), toInt3(dimsimage), NULL, nimages);
		CudaWriteToBinaryFile("d_masks.bin", d_masks, Elements2(dimsimage) * nimages * sizeof(tfloat));
		tfloat* d_maskedoriginal = CudaMallocValueFilled(Elements2(dimsimage) * nimages, 0.0f);
		cudaMemcpy(d_maskedoriginal, d_images, Elements2(dimsimage) * nimages * sizeof(tfloat), cudaMemcpyDeviceToDevice);
		d_NormMonolithic(d_images, d_maskedoriginal, Elements2(dimsimage), d_masks, T_NORM_MEAN01STD, nimages);
		d_MultiplyByVector(d_maskedoriginal, d_masks, d_maskedoriginal, Elements2(dimsimage) * nimages);
		CudaWriteToBinaryFile("d_maskedoriginal.bin", d_masks, Elements2(dimsimage) * nimages * sizeof(tfloat));
		tfloat* d_scores = CudaMallocValueFilled(nimages, 0.0f);
		tfloat* h_scores = (tfloat*)malloc(nimages * sizeof(tfloat));

		scores << "WBP:\n\n";
		for (int s = 1; s <= 3; s++)
		{
			d_RecWBP(d_volume, dimsvolume, volumeoffset, d_images, dimsimage, nimages, h_angles, h_translations, h_scales, T_INTERP_LINEAR, s, true);
			CudaWriteToBinaryFile("d_volumewbp.bin", d_volume, Elements(dimsvolume) * sizeof(tfloat));
			/*d_Pad(d_volume, d_volume2, dimsvolume, toInt3(64, 64, 16), T_PAD_VALUE, 0.0f);
			d_NormMonolithic(d_volume2, d_volume2, 64 * 64 * 16, T_NORM_MEAN01STD, 1);
			d_MultiplyByVector(d_volume2, d_volumeoriginal, d_volume2, 64 * 64 * 16);
			d_SumMonolithic(d_volume2, d_scores, 64 * 64 * 16, 1);
			cudaMemcpy(h_scores, d_scores, sizeof(tfloat), cudaMemcpyDeviceToHost);
			scores << h_scores[0] << "\n";
			continue;*/

			int3 dimsvolume2 = toInt3(64, 64, 64);
			d_Pad(d_volume, d_volume2, dimsvolume, dimsvolume2, T_PAD_VALUE, 0.0f);
			d_RemapFull2FullFFT(d_volume2, d_volume2, dimsvolume2);
			d_ProjForward(d_volume2, dimsvolume2, d_reprojections, toInt3(dimsimage), h_angles, T_INTERP_CUBIC, nimages);
			d_RemapFullFFT2Full(d_reprojections, d_reprojections, toInt3(dimsimage), nimages);
			//CudaWriteToBinaryFile("d_wbp.bin", d_reprojections, Elements2(dimsimage) * nimages * sizeof(tfloat));

			d_NormMonolithic(d_reprojections, d_reprojections, Elements2(dimsimage), d_masks, T_NORM_MEAN01STD, nimages);
			d_MultiplyByVector(d_reprojections, d_maskedoriginal, d_reprojections, Elements2(dimsimage) * nimages);
			d_SumMonolithic(d_reprojections, d_scores, Elements2(dimsimage), nimages);
			cudaMemcpy(h_scores, d_scores, nimages * sizeof(tfloat), cudaMemcpyDeviceToHost);
			double sum = 0.0f;
			for (int n = 0; n < nimages; n++)
				sum += h_scores[n];
			sum /= (double)nimages;
			scores << sum << "\n";
		}

		scores << "\nSIRT:\n\n";
		for (int s = 1; s <= 3; s++)
		{
			for (int i = 100; i <= 150; i += 10)
			{
				d_RecSIRT(d_volume, NULL, dimsvolume, volumeoffset, d_images, dimsimage, nimages, h_angles, h_translations, h_scales, T_INTERP_LINEAR, s, i, true);
				CudaWriteToBinaryFile("d_volumesirt.bin", d_volume, Elements(dimsvolume) * sizeof(tfloat));
				/*d_Pad(d_volume, d_volume2, dimsvolume, toInt3(64, 64, 16), T_PAD_VALUE, 0.0f);
				d_NormMonolithic(d_volume2, d_volume2, 64 * 64 * 16, T_NORM_MEAN01STD, 1);
				d_MultiplyByVector(d_volume2, d_volumeoriginal, d_volume2, 64 * 64 * 16);
				d_SumMonolithic(d_volume2, d_scores, 64 * 64 * 16, 1);
				cudaMemcpy(h_scores, d_scores, sizeof(tfloat), cudaMemcpyDeviceToHost);
				scores << h_scores[0] << "\n";
				continue;*/

				int3 dimsvolume2 = toInt3(64, 64, 64);
				d_Pad(d_volume, d_volume2, dimsvolume, dimsvolume2, T_PAD_VALUE, 0.0f);
				d_RemapFull2FullFFT(d_volume2, d_volume2, dimsvolume2);
				d_ProjForward(d_volume2, dimsvolume2, d_reprojections, toInt3(dimsimage), h_angles, T_INTERP_CUBIC, nimages);
				d_RemapFullFFT2Full(d_reprojections, d_reprojections, toInt3(dimsimage), nimages);
				//CudaWriteToBinaryFile("d_sirt.bin", d_reprojections, Elements2(dimsimage) * nimages * sizeof(tfloat));

				d_NormMonolithic(d_reprojections, d_reprojections, Elements2(dimsimage), d_masks, T_NORM_MEAN01STD, nimages);
				d_MultiplyByVector(d_reprojections, d_maskedoriginal, d_reprojections, Elements2(dimsimage) * nimages);
				d_SumMonolithic(d_reprojections, d_scores, Elements2(dimsimage), nimages);
				cudaMemcpy(h_scores, d_scores, nimages * sizeof(tfloat), cudaMemcpyDeviceToHost);
				double sum = 0.0f;
				for (int n = 0; n < nimages; n++)
					sum += h_scores[n];
				sum /= (double)nimages;
				scores << sum << "\n";
			}
			scores << "\n";
		}

		cudaFree(d_scores);
		cudaFree(d_masks);
		cudaFree(d_maskedoriginal);
		cudaFree(d_volume);
		cudaFree(d_images);
		cudaFree(d_reprojections);
		free(h_scales);
		free(h_translations);
		free(h_angles);
	}

	cudaDeviceReset();
}