#include "Prerequisites.h"

TEST(Correlation, Picker)
{
	cudaDeviceReset();

	//Case 1:
	//{
	//	HeaderMRC refheader = ReadMRCHeader("Data/Correlation/refs.mrc");
	//	int3 dimsref = refheader.dimensions;
	//	uint nrefs = 1;// dimsref.z;
	//	dimsref.z = 1;
	//	void* h_refraw;
	//	ReadMRC("Data/Correlation/refs.mrc", &h_refraw);
	//	tfloat* d_ref = MixedToDeviceTfloat(h_refraw, refheader.mode, Elements(dimsref) * nrefs);
	//	d_MultiplyByScalar(d_ref, d_ref, Elements(dimsref) * nrefs, -1.0f);

	//	tfloat* d_refmask = CudaMallocValueFilled(Elements(dimsref) * nrefs, (tfloat)1);
	//	tfloat maskradius = 74;
	//	d_SphereMask(d_refmask, d_refmask, dimsref, &maskradius, 1, NULL, nrefs);

	//	HeaderMRC imageheader = ReadMRCHeader("Data/Correlation/image.mrc");
	//	int3 dimsimage = imageheader.dimensions;
	//	void* h_imageraw;
	//	ReadMRC("Data/Correlation/image.mrc", &h_imageraw);
	//	tfloat* d_image = MixedToDeviceTfloat(h_imageraw, imageheader.mode, Elements(dimsimage));
	//	d_NormMonolithic(d_image, d_image, Elements(dimsimage), T_NORM_MEAN01STD, 1);

	//	tfloat* d_ctf = CudaMallocValueFilled(ElementsFFT(dimsimage), (tfloat)1);

	//	Picker picker;
	//	picker.Initialize(d_ref, dimsref, nrefs, d_refmask, true, true, dimsimage, 138);

	//	tfloat* d_bestccf = CudaMallocValueFilled(Elements(dimsimage), -1e30f);
	//	tfloat3* d_bestangle = (tfloat3*)CudaMallocValueFilled(Elements(dimsimage) * 3, (tfloat)0);
	//	int* d_bestref = CudaMallocValueFilled(Elements(dimsimage), -1);

	//	picker.SetImage(d_image, d_ctf);
	//	picker.PerformCorrelation(0, ToRad(5.0), d_bestccf, d_bestangle, d_bestref);

	//	d_WriteMRC(d_bestccf, dimsimage, "d_bestccf.mrc");
	//}

	//Case 2:
	//{
	//	HeaderMRC refheader = ReadMRCHeader("Data/Correlation/3dref.padded.mrc");
	//	int3 dimsref = refheader.dimensions;
	//	uint nrefs = 1;
	//	void* h_refraw;
	//	ReadMRC("Data/Correlation/3dref.padded.mrc", &h_refraw);
	//	tfloat* d_ref = MixedToDeviceTfloat(h_refraw, refheader.mode, Elements(dimsref));
	//	d_NormMonolithic(d_ref, d_ref, Elements(dimsref), T_NORM_MEAN01STD, nrefs);
	//	//d_MultiplyByScalar(d_ref, d_ref, Elements(dimsref), -1.0f);

	//	tfloat* d_refmask = CudaMallocValueFilled(Elements(dimsref), (tfloat)1);
	//	tfloat maskradius = 6;
	//	d_SphereMask(d_refmask, d_refmask, dimsref, &maskradius, 1, NULL, nrefs);

	//	HeaderMRC imageheader = ReadMRCHeader("Data/Correlation/3dimage.mrc");
	//	int3 dimsimage = imageheader.dimensions;
	//	void* h_imageraw;
	//	ReadMRC("Data/Correlation/3dimage.mrc", &h_imageraw);
	//	tfloat* d_image = MixedToDeviceTfloat(h_imageraw, imageheader.mode, Elements(dimsimage));
	//	d_NormMonolithic(d_image, d_image, Elements(dimsimage), T_NORM_MEAN01STD, 1);

	//	tfloat* d_ctf = CudaMallocValueFilled(ElementsFFT(dimsref), (tfloat)1);

	//	Picker picker;
	//	picker.Initialize(d_ref, dimsref, d_refmask, dimsimage);

	//	tfloat* d_bestccf = CudaMallocValueFilled(Elements(dimsimage), -1e30f);
	//	tfloat3* d_bestangle = (tfloat3*)CudaMallocValueFilled(Elements(dimsimage) * 3, (tfloat)0);

	//	picker.SetImage(d_image, d_ctf);
	//	picker.PerformCorrelation(ToRad(30.0), d_bestccf, d_bestangle);

	//	d_WriteMRC(d_bestccf, dimsimage, "d_bestccf.mrc");
	//}

	//Case 3:
	{
		HeaderMRC refheader = ReadMRCHeader("Data/Correlation/decoy5.mrc");
		int3 dimsref = refheader.dimensions;
		uint nrefs = 1;
		void* h_refraw;
		ReadMRC("Data/Correlation/decoy5.mrc", &h_refraw);
		tfloat* d_ref = MixedToDeviceTfloat(h_refraw, refheader.mode, Elements(dimsref));
		//d_NormMonolithic(d_ref, d_ref, Elements(dimsref), T_NORM_MEAN01STD, nrefs);
		d_MultiplyByScalar(d_ref, d_ref, Elements(dimsref), -1.0f);

		/*void* h_maskraw;
		ReadMRC("Data/Correlation/3dmask.padded.mrc", &h_maskraw);
		tfloat* d_mask = MixedToDeviceTfloat(h_maskraw, refheader.mode, Elements(dimsref));*/
		tfloat* d_mask = CudaMallocValueFilled(Elements(dimsref), (tfloat)1);

		HeaderMRC imageheader = ReadMRCHeader("Data/Correlation/pp.sc.mrc");
		int3 dimsimage = imageheader.dimensions;
		void* h_imageraw;
		ReadMRC("Data/Correlation/pp.sc.mrc", &h_imageraw);
		tfloat* d_image = MixedToDeviceTfloat(h_imageraw, imageheader.mode, Elements(dimsimage));
		//d_NormMonolithic(d_image, d_image, Elements(dimsimage), T_NORM_MEAN01STD, 1);

		void* h_ctfraw;
		ReadMRC("Data/Correlation/ribo.psf.mrc", &h_ctfraw);
		tfloat* d_ctf = MixedToDeviceTfloat(h_ctfraw, refheader.mode, ElementsFFT(dimsref));

		Picker picker;
		picker.Initialize(d_ref, dimsref, d_mask, dimsimage);

		tfloat* d_bestccf = CudaMallocValueFilled(Elements(dimsimage), -1e30f);
		tfloat3* d_bestangle = (tfloat3*)CudaMallocValueFilled(Elements(dimsimage) * 3, (tfloat)0);

		picker.SetImage(d_image, d_ctf);
		picker.PerformCorrelation(ToRad(15.0), d_bestccf, d_bestangle);

		d_WriteMRC(d_bestccf, dimsimage, "d_bestccf.mrc");
	}

	cudaDeviceReset();
}