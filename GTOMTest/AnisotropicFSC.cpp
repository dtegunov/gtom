#include "Prerequisites.h"

TEST(Resolution, AnisotropicFSC)
{
	cudaDeviceReset();

	//Case 1:
	{
		HeaderMRC header = ReadMRCHeader("Data\\Resolution\\half1.mrc");
		int3 dimsvolume = header.dimensions;
		uint nvolumes = dimsvolume.z / dimsvolume.x;
		dimsvolume.z = dimsvolume.x;

		void* h_mrcraw1, *h_mrcraw2;
		ReadMRC("Data\\Resolution\\half1.mrc", &h_mrcraw1);
		ReadMRC("Data\\Resolution\\half2.mrc", &h_mrcraw2);
		tfloat* d_input1 = MixedToDeviceTfloat(h_mrcraw1, header.mode, Elements(dimsvolume) * nvolumes);
		tfloat* d_input2 = MixedToDeviceTfloat(h_mrcraw2, header.mode, Elements(dimsvolume) * nvolumes);

		int2 anglesteps = toInt2(3, 2);
		tfloat* d_resolution = (tfloat*)CudaMallocValueFilled(Elements2(anglesteps), (tfloat)0);

		d_AnisotropicFSCMap(d_input1, d_input2, dimsvolume, d_resolution, anglesteps, dimsvolume.x / 2, T_FSC_THRESHOLD, (tfloat)0.143, NULL, 1);

		d_WriteMRC(d_resolution, toInt3(anglesteps), "d_anisotropic.mrc");

		cudaFree(d_resolution);
		cudaFree(d_input2);
		cudaFree(d_input1);

		cudaFreeHost(h_mrcraw1);
		cudaFreeHost(h_mrcraw2);
	}

	cudaDeviceReset();
}