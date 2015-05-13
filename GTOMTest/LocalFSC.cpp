#include "Prerequisites.h"

TEST(Resolution, LocalFSC)
{
	cudaDeviceReset();

	//Case 1:
	{
		HeaderMRC header = ReadMRCHeader("Data\\Resolution\\half1.mrc");
		int3 dimsvolume = header.dimensions;
		uint nvolumes = dimsvolume.z / dimsvolume.x;
		dimsvolume.z = dimsvolume.x;

		int windowsize = 40;
		int shells = windowsize / 2;

		void* h_mrcraw1, *h_mrcraw2;
		ReadMRC("Data\\Resolution\\half1.mrc", &h_mrcraw1);
		ReadMRC("Data\\Resolution\\half2.mrc", &h_mrcraw2);
		tfloat* d_input1 = MixedToDeviceTfloat(h_mrcraw1, header.mode, Elements(dimsvolume) * nvolumes);
		tfloat* d_input2 = MixedToDeviceTfloat(h_mrcraw2, header.mode, Elements(dimsvolume) * nvolumes);

		tfloat* d_resolution = (tfloat*)CudaMallocValueFilled(Elements(dimsvolume), (tfloat)0);
		
		d_LocalFSC(d_input1, d_input2, dimsvolume, nvolumes, d_resolution, windowsize, shells, (tfloat)0.143);

		d_WriteMRC(d_resolution, dimsvolume, "d_resolution.mrc");
		
		cudaFree(d_resolution);
		cudaFree(d_input2);
		cudaFree(d_input1);

		cudaFreeHost(h_mrcraw1);
		cudaFreeHost(h_mrcraw2);
	}

	cudaDeviceReset();
}