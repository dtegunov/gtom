#include "Prerequisites.h"

TEST(Correlation, Picker)
{
	cudaDeviceReset();

	//Case 1:
	{
		HeaderMRC refheader = ReadMRCHeader("Data/Correlation/refs.mrc");
		int3 dimsref = refheader.dimensions;
		uint nrefs = dimsref.z;
		dimsref.z = 1;
		void* h_refraw;
		ReadMRC("Data/Correlation/refs.mrc", &h_refraw);
		tfloat* d_ref = MixedToDeviceTfloat(h_refraw, refheader.mode, Elements(dimsref) * nrefs);
		d_MultiplyByScalar(d_ref, d_ref, Elements(dimsref) * nrefs, -1.0f);

		tfloat* d_refmask = CudaMallocValueFilled(Elements(dimsref) * nrefs, (tfloat)1);
		tfloat maskradius = 74;
		d_SphereMask(d_refmask, d_refmask, dimsref, &maskradius, 4, NULL, nrefs);

		HeaderMRC imageheader = ReadMRCHeader("Data/Correlation/image.mrc");
		int3 dimsimage = imageheader.dimensions;
		void* h_imageraw;
		ReadMRC("Data/Correlation/image.mrc", &h_imageraw);
		tfloat* d_image = MixedToDeviceTfloat(h_imageraw, imageheader.mode, Elements(dimsimage));

		tfloat* d_ctf = CudaMallocValueFilled(ElementsFFT(dimsimage), (tfloat)1);

		Picker picker;
		picker.Initialize(d_ref, dimsref, nrefs, d_refmask, false, true, dimsimage, 138);

		tfloat* d_bestccf = CudaMallocValueFilled(Elements(dimsimage), -1e30f);
		tfloat3* d_bestangle = (tfloat3*)CudaMallocValueFilled(Elements(dimsimage) * 3, (tfloat)0);
		int* d_bestref = CudaMallocValueFilled(Elements(dimsimage), -1);

		picker.SetImage(d_image, d_ctf);
		picker.PerformCorrelation(1, ToRad(15.0), d_bestccf, d_bestangle, d_bestref);
	}

	cudaDeviceReset();
}