#include "Prerequisites.h"

TEST(CTF, TiltFit)
{
	cudaDeviceReset();

	//Spectra accumulation:
	{
		tfloat* d_input = (tfloat*)CudaMallocFromBinaryFile("Data/CTF/Input_Accumulate.bin");

		CTFFitParams fp;
		fp.dimsperiodogram = toInt2(1024, 1024);
		fp.maskouterradius = fp.dimsperiodogram.x / 2;
		fp.maskinnerradius = 0;
		uint relevantlength = fp.maskouterradius - fp.maskinnerradius;

		CTFParams p;
		p.pixelsize = 1.35e-10;
		p.defocus = -1.0e-6;

		uint nspectra = 2;
		tfloat h_defoci[2] { 0.0e-6f, -2.0e-6f };
		tfloat accumulateddefocus = 0;
		tfloat offset = -1.0e-6f;

		tfloat* d_defoci = (tfloat*)CudaMallocFromHostArray(h_defoci, nspectra * sizeof(tfloat));
		tfloat* d_defocusoffsets = (tfloat*)CudaMallocFromHostArray(&offset, sizeof(tfloat));
		tfloat* d_accumulated = CudaMallocValueFilled(relevantlength, (tfloat)0);

		d_AccumulateSpectra(d_input, d_defoci, nspectra, d_accumulated, accumulateddefocus, d_defocusoffsets, p, fp);

		CudaWriteToBinaryFile("d_accumulated.bin", d_accumulated, relevantlength * sizeof(tfloat));
	}

	cudaDeviceReset();

	//Case 1:
	{
		HeaderMRC header = ReadMRCHeader("Data/CTF/L3Tomo3_plus.st");
		int2 dimsimage = toInt2(header.dimensions.x, header.dimensions.y);
		uint nimages = header.dimensions.z;
		nimages = 1;
		int2 dimsregion = toInt2(256, 256);
		void* h_mrcraw;
		ReadMRC("Data/CTF/L3Tomo3_plus.st", (void**)&h_mrcraw);
		tfloat* h_images = MixedToHostTfloat(h_mrcraw, header.mode, Elements(header.dimensions));
		cudaFreeHost(h_mrcraw);

		CTFFitParams fp;
		fp.maskinnerradius = 24;
		fp.maskouterradius = 72;
		fp.dimsperiodogram = dimsregion;
		fp.defocus = tfloat3(-3.0e-6f, 3.0e-6f, 0.1e-6f);

		vector<CTFTiltParams> v_startparams;
		for (uint i = 0; i < nimages; i++)
		{
			CTFParams params;
			params.pixelsize = 3.42e-10;
			params.defocus = -5.6e-6;
			params.astigmatismangle = ToRad(0);
			params.defocusdelta = 0.0e-6f;

			CTFTiltParams tp(tfloat3(ToRad(-96.9174f), ToRad(i * (2.0f)), 0.0f), params);
			v_startparams.push_back(tp);
		}

		tfloat2 specimentilt = tfloat2(2.73189187, 0.100185998);
		tfloat* h_defoci = MallocValueFilled(nimages, (tfloat)0);

		h_CTFTiltFit(h_images, dimsimage, nimages, 0.5f, v_startparams, fp, ToRad(15.0f), specimentilt, h_defoci);

		free(h_defoci);
		cudaFreeHost(h_images);

		CTFTiltParams tilt(0, tfloat2(ToRad(-96.9174f), 0), specimentilt, CTFParams());
		int2 dimsgrid;
		int3* h_positions = GetEqualGridSpacing(dimsimage, dimsregion, 0.75f, dimsgrid);
		tfloat* h_zvalues = (tfloat*)malloc(Elements2(dimsgrid) * sizeof(tfloat));
		tilt.GetZGrid2D(dimsimage, dimsregion, h_positions, Elements2(dimsgrid), h_zvalues);

		WriteToBinaryFile("d_zvalues.bin", h_zvalues, Elements2(dimsgrid) * sizeof(tfloat));

		free(h_positions);
		free(h_zvalues);
	}

	cudaDeviceReset();
}