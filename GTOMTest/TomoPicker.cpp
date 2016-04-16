#include "Prerequisites.h"

TEST(Correlation, TomoPicker)
{
	cudaDeviceReset();

	//Case 1:
	{
		HeaderMRC refheader = ReadMRCHeader("Data/Correlation/3dref.mrc");
		int3 dimsref = refheader.dimensions;
		uint nrefs = 1;
		void* h_refraw;
		ReadMRC("Data/Correlation/3dref.mrc", &h_refraw);
		tfloat* d_ref = MixedToDeviceTfloat(h_refraw, refheader.mode, Elements(dimsref));
		//d_NormMonolithic(d_ref, d_ref, Elements(dimsref), T_NORM_MEAN01STD, nrefs);
		d_MultiplyByScalar(d_ref, d_ref, Elements(dimsref), -1.0f);

		void* h_refmaskraw;
		ReadMRC("Data/Correlation/3dmask.mrc", &h_refmaskraw);
		tfloat* d_refmask = MixedToDeviceTfloat(h_refmaskraw, refheader.mode, Elements(dimsref));

		HeaderMRC imageheader = ReadMRCHeader("Data/Correlation/t2.scaled.ali");
		int3 dimsimage = imageheader.dimensions;
		void* h_imageraw;
		ReadMRC("Data/Correlation/t2.scaled.ali", &h_imageraw);
		tfloat* d_image = MixedToDeviceTfloat(h_imageraw, imageheader.mode, Elements(dimsimage));
		//d_NormMonolithic(d_image, d_image, Elements(dimsimage), T_NORM_MEAN01STD, nimages);
		int nimages = dimsimage.z;
		dimsimage.z = 1;
		
		TomoPicker picker;
		picker.Initialize(d_ref, dimsref, d_refmask, false, toInt2(dimsimage), nimages);

		double tilts[61] = { -50.5,
			-48.4,
			-46.4,
			-44.3,
			-42.2,
			-40.2,
			-38.1,
			-36.1,
			-34.1,
			-32.1,
			-30.1,
			-28.1,
			-26.1,
			-24.1,
			-22.1,
			-20.1,
			-18.1,
			-16.1,
			-14.1,
			-12.1,
			-10.1,
			-8.1,
			-6.1,
			-4.1,
			-2.1,
			0.0,
			2.0,
			4.1,
			6.3,
			8.7,
			10.9,
			13.3,
			15.3,
			17.0,
			18.7,
			20.6,
			22.4,
			24.3,
			26.2,
			28.2,
			30.2,
			32.0,
			34.0,
			35.9,
			37.9,
			39.9,
			41.9,
			43.8,
			45.8,
			47.8,
			49.8,
			51.8,
			53.8,
			55.8,
			57.7,
			59.7,
			61.7,
			63.7,
			65.7,
			67.7,
			69.7
		};
		tfloat3* h_angles = (tfloat3*)malloc(nimages * sizeof(tfloat3));
		for (int i = 0; i < nimages; i++)
			h_angles[i] = tfloat3(0.0f, -(tfloat)ToRad(tilts[i]), 0.0f);

		tfloat* h_imageweights = MallocValueFilled(nimages, (tfloat)1);
		for (int i = 0; i < nimages; i++)
			h_imageweights[i] = 1 / (1 + tan(h_angles[i].y) * tan(h_angles[i].y));

		picker.SetImage(d_image, h_angles, h_imageweights);

		int3 dimstomo = toInt3(800, 800, 250);

		tfloat* d_bestccf = CudaMallocValueFilled(Elements(dimstomo), -1e30f);
		tfloat3* d_bestangle = (tfloat3*)CudaMallocValueFilled(Elements(dimstomo) * 3, (tfloat)0);
		
		picker.PerformCorrelation(d_bestccf, d_bestangle, dimstomo, ToRad(12.5));

		d_WriteMRC(d_bestccf, dimstomo, "d_bestccf.mrc");
	}

	cudaDeviceReset();
}