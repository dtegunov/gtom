#include "Prerequisites.h"

TEST(Reconstruction, RecWBP)
{
	cudaDeviceReset();

	//Case 1:
	{
		int3 dimsvolume = toInt3(32, 32, 32);
		tfloat3 volumeoffset = tfloat3(0.0f, 0.0f, 0.0f);
		int2 dimsimage = toInt2(32, 32);
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
		tfloat* d_images = CudaMallocValueFilled(Elements2(dimsimage) * nimages, 0.0f);

		tfloat* h_volume = MallocValueFilled(Elements(dimsvolume), 0.0f);
		h_volume[(dimsvolume.z / 2 * dimsvolume.y + dimsvolume.y / 2) * dimsvolume.x + dimsvolume.x / 2] = 1.0f;
		cudaMemcpy(d_volume, h_volume, Elements(dimsvolume) * sizeof(tfloat), cudaMemcpyHostToDevice);

		tfloat* h_images = MallocValueFilled(Elements2(dimsimage) * nimages, 0.0f);
		for (int n = 0; n < nimages; n++)
			h_images[Elements2(dimsimage) * n + dimsimage.y / 2 * dimsimage.x + dimsimage.x / 2] = 1.0f;
		cudaMemcpy(d_images, h_images, Elements2(dimsimage) * nimages * sizeof(tfloat), cudaMemcpyHostToDevice);

		d_RecWBP(d_volume, dimsvolume, volumeoffset, d_images, dimsimage, nimages, h_angles, h_translations, h_scales, NULL, T_INTERP_CUBIC, true);

		CudaWriteToBinaryFile("d_images.bin", d_images, Elements2(dimsimage) * nimages * sizeof(tfloat));
		CudaWriteToBinaryFile("d_volume.bin", d_volume, Elements(dimsvolume) * sizeof(tfloat));

		cudaFree(d_volume);
		cudaFree(d_images);
		free(h_scales);
		free(h_translations);
		free(h_angles);
	}

	cudaDeviceReset();
}