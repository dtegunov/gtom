#include "Prerequisites.h"

TEST(Resolution, LocalFSCBfac)
{
	cudaDeviceReset();

	//Case 1:
	{
		cudaSetDevice(1);

		HeaderMRC header = ReadMRCHeader("F:\\badaben\\vlion123\\WarpedAlign2\\C2\\run1_half1_class001_unfil.mrc");
		int3 dimsvolume = header.dimensions;
		uint nvolumes = 1;
		dimsvolume.z = dimsvolume.x;

		int windowsize = 18;

		void* h_mrcraw1, *h_mrcraw2;
		ReadMRC("F:\\badaben\\vlion123\\WarpedAlign2\\C2\\run1_half1_class001_unfil.mrc", &h_mrcraw1);
		ReadMRC("F:\\badaben\\vlion123\\WarpedAlign2\\C2\\run1_half2_class001_unfil.mrc", &h_mrcraw2);
		tfloat* d_input1 = MixedToDeviceTfloat(h_mrcraw1, header.mode, Elements(dimsvolume) * nvolumes);
		tfloat* d_input2 = MixedToDeviceTfloat(h_mrcraw2, header.mode, Elements(dimsvolume) * nvolumes);

		tfloat* d_resolution = CudaMallocValueFilled(Elements(dimsvolume), (tfloat)0);
		tfloat* d_bfac = CudaMallocValueFilled(Elements(dimsvolume), (tfloat)0);
		tfloat* d_corrected = CudaMallocValueFilled(Elements(dimsvolume), (tfloat)0);
		tfloat* d_unsharpened = CudaMallocValueFilled(Elements(dimsvolume), (tfloat)0);
		tfloat* d_normalized = CudaMallocValueFilled(Elements(dimsvolume), (tfloat)0);

		float Pixelsize = 3.42f; // header.pixelsize.x / header.dimensions.x;

		d_LocalFSCBfac(d_input1, d_input2, dimsvolume, d_resolution, d_bfac, d_corrected, d_unsharpened, windowsize, 0.3f, 15, Pixelsize, 0, 0, -1.34f, false);

		tfloat* d_stdunsharp;
		cudaMalloc((void**)&d_stdunsharp, Elements(dimsvolume) * sizeof(tfloat));
		tfloat* d_stdsharp;
		cudaMalloc((void**)&d_stdsharp, Elements(dimsvolume) * sizeof(tfloat));

		d_LocalStd(d_unsharpened, dimsvolume, 3, d_stdunsharp);
		d_LocalStd(d_corrected, dimsvolume, 3, d_stdsharp);

		d_DivideSafeByVector(d_stdunsharp, d_stdsharp, d_stdunsharp, Elements(dimsvolume));
		d_MultiplyByVector(d_corrected, d_stdunsharp, d_normalized, Elements(dimsvolume));

		d_WriteMRC(d_resolution, dimsvolume, "F:\\badaben\\vlion123\\WarpedAlign2\\C2\\resolution_output.mrc");
		//d_WriteMRC(d_bfac, dimsvolume, "F:\\stefanribo\\vlion\\MiniWarpedLocalOST\\bee\\bfac_aniso.mrc");
		d_WriteMRC(d_corrected, dimsvolume, "F:\\badaben\\vlion123\\WarpedAlign2\\C2\\localfilt.mrc");
		//d_WriteMRC(d_unsharpened, dimsvolume, "F:\\stefanribo\\vlion\\MiniWarpedLocalOST\\bee\\unsharpened_aniso.mrc");
		//d_WriteMRC(d_normalized, dimsvolume, "F:\\stefanribo\\vlion\\MiniWarpedLocalOST\\bee\\normalized_aniso.mrc");

		cudaFree(d_stdsharp);
		cudaFree(d_stdunsharp);
		cudaFree(d_resolution);
		cudaFree(d_bfac);
		cudaFree(d_input2);
		cudaFree(d_input1);

		cudaFreeHost(h_mrcraw1);
		cudaFreeHost(h_mrcraw2);
	}

	cudaDeviceReset();
}