#include "Prerequisites.cuh"
#include "FFT.cuh"
#include "Generics.cuh"
#include "Helper.cuh"
#include "Projection.cuh"
#include "Reconstruction.cuh"
#include "Transformation.cuh"


////////////////////////////////////////////////////////////
//Performs 3D reconstruction using Weighted Backprojection//
////////////////////////////////////////////////////////////

void d_RecWBP(tfloat* d_volume, int3 dimsvolume, tfloat3 offsetfromcenter, tfloat* d_image, int2 dimsimage, int nimages, tfloat3* h_angles, tfloat2* h_offsets, tfloat2* h_scales, T_INTERP_MODE mode, bool outputzerocentered)
{
	int3 dimspadded = toInt3(dimsimage.x * 1, dimsimage.y * 1, 1);

	/*tfloat* d_paddedimage;
	cudaMalloc((void**)&d_paddedimage, Elements2(dimspadded) * nimages * sizeof(tfloat));
	d_Pad(d_image, d_paddedimage, toInt3(dimsimage), dimspadded, T_PAD_VALUE, 0.0f, nimages);*/

	int* h_indices = (int*)malloc(nimages * sizeof(int));
	for (int n = 0; n < nimages; n++)
		h_indices[n] = n;

	tcomplex* d_imageft;
	cudaMalloc((void**)&d_imageft, ElementsFFT(dimspadded) * nimages * sizeof(tcomplex));

	tfloat* d_weighted;
	cudaMalloc((void**)&d_weighted, Elements2(dimsimage) * nimages * sizeof(tfloat));

	size_t memlimit = 512 << 20;
	int ftbatch = memlimit / (Elements2(dimsimage) * sizeof(tfloat) * 6);

	for (int b = 0; b < nimages; b += ftbatch)
		d_FFTR2C(d_image + Elements2(dimsimage) * b, d_imageft + ElementsFFT2(dimsimage) * b, 2, dimspadded, min(nimages - b, ftbatch));
	d_Exact2DWeighting(d_imageft, toInt2(dimspadded), h_indices, h_angles, nimages, dimspadded.x, false, nimages);
	for (int b = 0; b < nimages; b += ftbatch)
		d_IFFTC2R(d_imageft + ElementsFFT2(dimsimage) * b, d_weighted + Elements2(dimsimage) * b, 2, dimspadded, min(nimages - b, ftbatch));
	free(h_indices);
	cudaFree(d_imageft);

	//d_Pad(d_paddedimage, (tfloat*)d_imageft, dimspadded, toInt3(dimsimage), T_PAD_VALUE, 0.0f, nimages);

	d_ValueFill(d_volume, Elements(dimsvolume), 0.0f);
	d_ProjBackward(d_volume, dimsvolume, offsetfromcenter, d_weighted, dimsimage, h_angles, h_offsets, h_scales, mode, outputzerocentered, nimages);
	
	cudaFree(d_weighted);
}