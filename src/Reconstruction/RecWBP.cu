#include "Prerequisites.cuh"
#include "FFT.cuh"
#include "Generics.cuh"
#include "Helper.cuh"
#include "Projection.cuh"
#include "Transformation.cuh"


////////////////////////////////////////////////////////////
//Performs 3D reconstruction using Weighted Backprojection//
////////////////////////////////////////////////////////////

void d_RecWBP(tfloat* d_volume, int3 dimsvolume, tfloat3 offsetfromcenter, tfloat* d_image, int2 dimsimage, int nimages, tfloat3* h_angles, tfloat2* h_offsets, tfloat2* h_scales, T_INTERP_MODE mode, int supersample, bool outputzerocentered)
{
	dimsimage = toInt2(dimsimage.x * supersample, dimsimage.y * supersample);
	dimsvolume = toInt3(dimsvolume.x * supersample, dimsvolume.y * supersample, dimsvolume.z * supersample);

	tfloat* d_superimage, *d_supervolume;

	cudaMalloc((void**)&d_superimage, Elements2(dimsimage) * nimages * sizeof(tfloat));
	if (supersample > 1)
		cudaMalloc((void**)&d_supervolume, Elements(dimsvolume) * sizeof(tfloat));
	else
		d_supervolume = d_volume;
	d_Pad(d_image, d_superimage, toInt3(dimsimage.x / supersample, dimsimage.y / supersample, 1), toInt3(dimsimage), T_PAD_VALUE, 0.0f, nimages);
	//CudaWriteToBinaryFile("d_superimage.bin", d_superimage, Elements2(dimsimage) * nimages * sizeof(tfloat));

	int* h_indices = (int*)malloc(nimages * sizeof(int));
	for (int n = 0; n < nimages; n++)
		h_indices[n] = n;

	tcomplex* d_imageft;
	cudaMalloc((void**)&d_imageft, ElementsFFT2(dimsimage) * nimages * sizeof(tcomplex));

	d_FFTR2C(d_superimage, d_imageft, 2, toInt3(dimsimage), nimages);
	d_Exact2DWeighting(d_imageft, dimsimage, h_indices, h_angles, nimages, dimsimage.x / 2, false, nimages);
	d_IFFTC2R(d_imageft, d_superimage, 2, toInt3(dimsimage), nimages);
	cudaFree(d_imageft);
	free(h_indices);
	//CudaWriteToBinaryFile("d_superimage.bin", d_superimage, Elements2(dimsimage) * nimages * sizeof(tfloat));

	tfloat* d_tempimage;
	cudaMalloc((void**)&d_tempimage, Elements2(toInt2(dimsimage.x / supersample, dimsimage.y / supersample)) * nimages * sizeof(tfloat));
	d_Pad(d_superimage, d_tempimage, toInt3(dimsimage), toInt3(dimsimage.x / supersample, dimsimage.y / supersample, 1), T_PAD_VALUE, 0.0f, nimages);
	d_Scale(d_tempimage, d_superimage, toInt3(dimsimage.x / supersample, dimsimage.y / supersample, 1), toInt3(dimsimage), T_INTERP_FOURIER, NULL, NULL, nimages);
	cudaFree(d_tempimage);
	//CudaWriteToBinaryFile("d_superimage.bin", d_superimage, Elements2(dimsimage) * nimages * sizeof(tfloat));

	d_ValueFill(d_supervolume, Elements(dimsvolume), 0.0f);
	d_ProjBackward(d_supervolume, dimsvolume, offsetfromcenter, d_superimage, toInt3(dimsimage), h_angles, h_offsets, h_scales, mode, outputzerocentered, nimages);

	if (supersample > 1)
	{
		d_Scale(d_supervolume, d_volume, dimsvolume, toInt3(dimsvolume.x / supersample, dimsvolume.y / supersample, dimsvolume.z / supersample), T_INTERP_FOURIER);
	}

	cudaFree(d_superimage);
	if (supersample > 1)
	{
		cudaFree(d_supervolume);
	}
}