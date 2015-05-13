#include "Prerequisites.cuh"
#include "Angles.cuh"
#include "CTF.cuh"
#include "DeviceFunctions.cuh"
#include "FFT.cuh"
#include "Generics.cuh"
#include "Helper.cuh"
#include "Masking.cuh"
#include "Reconstruction.cuh"


////////////////////////////
//CUDA kernel declarations//
////////////////////////////

__global__ void PrecomputeBlobKernel(tfloat* d_precompblob, uint dim, tfloat* d_funcvals, tfloat funcsampling, int funcelements, tfloat normftblob);
__global__ void SoftMaskKernel(tfloat* d_input, uint dim, uint dimft, uint n);
__global__ void UpdateWeightKernel(tcomplex* d_conv, tfloat* d_weight, tfloat* d_newweight, uint n);
__global__ void CorrectGriddingKernel(tfloat* d_volume, uint dim);


//////////////////////////////////////////////////////
//Performs 3D reconstruction using Fourier inversion//
//////////////////////////////////////////////////////

void d_ReconstructFourier(tcomplex* d_imagesft, tfloat* d_imagespsf, tcomplex* d_volumeft, tfloat* d_volumepsf, int3 dims, tfloat3* h_angles, tfloat2* h_shifts, int nimages, bool performgridding, bool everythingcentered)
{
	int3 dimsimage = toInt3(dims.x, dims.y, 1);

	if (!everythingcentered)	// d_imageft needs to be centered for reconstruction
		d_RemapHalfFFT2Half(d_imagesft, d_imagesft, dimsimage, nimages);

	d_ValueFill(d_volumeft, ElementsFFT(dims), make_cuComplex(0, 0));
	d_ValueFill(d_volumepsf, ElementsFFT(dims), (tfloat)0);

	d_ReconstructFourierAdd(d_volumeft, d_volumepsf, dims, d_imagesft, d_imagespsf, h_angles, h_shifts, nimages);

	tfloat* d_weights;
	cudaMalloc((void**)&d_weights, ElementsFFT(dims) * sizeof(tfloat));
	cudaMemcpy(d_weights, d_volumepsf, ElementsFFT(dims) * sizeof(tfloat), cudaMemcpyDeviceToDevice);

	d_MinOp(d_volumepsf, (tfloat)1, d_volumepsf, ElementsFFT(dims));
	d_MaxOp(d_weights, (tfloat)1, d_weights, ElementsFFT(dims));
	d_Inv(d_weights, d_weights, ElementsFFT(dims));
	d_ComplexMultiplyByVector(d_volumeft, d_weights, d_volumeft, ElementsFFT(dims));

	if (!everythingcentered)	// Volume and PSF come centered from d_ReconstructFourierAdd
	{
		d_RemapHalf2HalfFFT(d_volumeft, d_volumeft, dims);
		d_RemapHalf2HalfFFT(d_volumepsf, d_volumepsf, dims);
	}

	cudaFree(d_weights);

	if (!everythingcentered)
		d_RemapHalf2HalfFFT(d_imagesft, d_imagesft, dimsimage, nimages);
}

void d_ReconstructFourierPrecise(tfloat* d_images, tfloat* d_imagespsf, tfloat* d_volume, tfloat* d_volumepsf, int3 dims, tfloat3* h_angles, tfloat2* h_shifts, int nimages, bool dogridding)
{
	tcomplex* d_volumeft = (tcomplex*)CudaMallocValueFilled(ElementsFFT(dims) * 2, (tfloat)0);

	d_ReconstructFourierPreciseAdd(d_volumeft, d_volumepsf, dims, d_images, d_imagespsf, h_angles, h_shifts, nimages, T_INTERP_SINC, false, !dogridding);

	if (dogridding)
	{
		tfloat* d_newweight = CudaMallocValueFilled(Elements(dims), (tfloat)1);
		int TpB = min(192, NextMultipleOf(ElementsFFT(dims), 32));
		dim3 grid = dim3((ElementsFFT(dims) + TpB - 1) / TpB);
		SoftMaskKernel <<<grid, TpB>>> (d_newweight, dims.x, dims.x / 2 + 1, ElementsFFT(dims));
		/*d_SphereMask(d_newweight, d_newweight, dims, NULL, 0, NULL);
		d_RemapFull2HalfFFT(d_newweight, d_newweight, dims);*/

		d_ReconstructionFourierCorrection(d_volumepsf, d_newweight, dims, 2);
		CudaWriteToBinaryFile("d_newweight.bin", d_newweight, ElementsFFT(dims) * sizeof(tfloat));

		d_ComplexMultiplyByVector(d_volumeft, d_newweight, d_volumeft, ElementsFFT(dims));
		cudaFree(d_newweight);
	}

	d_IFFTC2R(d_volumeft, d_volume, 3, dims);
	d_RemapFullFFT2Full(d_volume, d_volume, dims);
	d_RemapHalfFFT2Half(d_volumepsf, d_volumepsf, dims);

	if (dogridding)
	{
		dim3 TpB = dim3(8, 8, 8);
		dim3 grid = dim3((dims.x + 7) / 8, (dims.x + 7) / 8, (dims.x + 7) / 8);
		CorrectGriddingKernel <<<grid, TpB>>> (d_volume, dims.x);
	}

	cudaFree(d_volumeft);
}

void d_ReconstructionFourierCorrection(tfloat* d_weight, tfloat* d_newweight, int3 dims, int paddingfactor)
{
	tfloat* d_zeroIm;
	cudaMalloc((void**)&d_zeroIm, ElementsFFT(dims) * sizeof(tfloat));
	tcomplex* d_conv;
	cudaMalloc((void**)&d_conv, ElementsFFT(dims) * sizeof(tcomplex));

	tfloat* d_buffer;
	cudaMalloc((void**)&d_buffer, ElementsFFT(dims) * sizeof(tfloat));

	// Precalc blob values
	tfloat* d_precompblob;
	{
		double radius = 1.9 * (double)paddingfactor;
		double alpha = 15.0;
		int order = 0;
		int elements = 10000;
		double sampling = 0.5 / (double)elements;
		tfloat* h_blobvalues = (tfloat*)malloc(elements * sizeof(tfloat));
		for (int i = 0; i < elements; i++)
			h_blobvalues[i] = kaiser_Fourier_value((double)i * sampling, radius, alpha, order);
		tfloat* d_blobvalues = (tfloat*)CudaMallocFromHostArray(h_blobvalues, elements * sizeof(tfloat));

		cudaMalloc((void**)&d_precompblob, Elements(dims) * sizeof(tfloat));
		int TpB = min(192, NextMultipleOf(Elements(dims), 32));
		dim3 grid = (Elements(dims) + TpB - 1) / TpB;
		PrecomputeBlobKernel <<<grid, TpB>>> (d_precompblob, dims.x, d_blobvalues, (tfloat)sampling, elements, h_blobvalues[0]);

		cudaFree(d_blobvalues);
		free(h_blobvalues);
	}

	for (int i = 0; i < 10; i++)
	{
		d_ValueFill(d_zeroIm, ElementsFFT(dims), (tfloat)0);

		d_MultiplyByVector(d_newweight, d_weight, d_buffer, ElementsFFT(dims));
		d_ConvertSplitComplexToTComplex(d_buffer, d_zeroIm, d_conv, ElementsFFT(dims));

		d_IFFTC2R(d_conv, (tfloat*)d_conv, 3, dims);

		d_MultiplyByVector((tfloat*)d_conv, d_precompblob, (tfloat*)d_conv, Elements(dims));

		d_FFTR2C((tfloat*)d_conv, d_conv, 3, dims);

		int TpB = min(192, NextMultipleOf(ElementsFFT(dims), 32));
		dim3 grid = dim3((ElementsFFT(dims) + TpB - 1) / TpB);
		UpdateWeightKernel <<<grid, TpB>>> (d_conv, d_weight, d_newweight, ElementsFFT(dims));
	}

	cudaFree(d_zeroIm);
	cudaFree(d_conv);
	cudaFree(d_buffer);
	cudaFree(d_precompblob);
}

__global__ void PrecomputeBlobKernel(tfloat* d_precompblob, uint dim, tfloat* d_funcvals, tfloat funcsampling, int funcelements, tfloat normftblob)
{
	uint id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= dim * dim * dim)
		return;

	uint idx = id % dim;
	uint idy = (id / dim) % dim;
	uint idz = id / (dim * dim);

	int x = (idx + dim / 2) % dim;
	x -= (int)(dim / 2);
	int y = (idy + dim / 2) % dim;
	y -= (int)(dim / 2);
	int z = (idz + dim / 2) % dim;
	z -= (int)(dim / 2);

	tfloat val = 0;
	tfloat r = sqrt((tfloat)(x * x + y * y + z * z)) / (tfloat)dim;
	int funcid = (int)(r / funcsampling);
	if (funcid < funcelements)
		val = d_funcvals[funcid] / normftblob;

	d_precompblob[id] = val;
}

__global__ void SoftMaskKernel(tfloat* d_input, uint dim, uint dimft, uint n)
{
	uint id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= n)
		return;

	uint idx = id % dimft;
	uint idy = (id / dimft) % dim;
	uint idz = id / (dimft * dim);

	int x = idx;
	int y = (idy + dim / 2) % dim;
	y -= (int)(dim / 2);
	int z = (idz + dim / 2) % dim;
	z -= (int)(dim / 2);

	/*float r = sqrt((float)(x * x + y * y + z * z));
	float falloff = min(max(0.0f, r - (float)(dim / 2 - 4)), 4.0f);
	falloff = cos(falloff * 0.25f * PI) * 0.5f + 0.5f;*/
	float falloff = x * x + y * y + z * z >= dim * dim / 4 ? 0.0f : 1.0f;
	d_input[id] *= falloff;
}

__global__ void UpdateWeightKernel(tcomplex* d_conv, tfloat* d_weight, tfloat* d_newweight, uint n)
{
	uint id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= n)
		return;

	tfloat update = abs(d_conv[id].x);
	tfloat weight = d_newweight[id];
	//if (d_weight[id] == 0)
		//d_newweight[id] = 0;
	if (update > 0)
		d_newweight[id] = min((tfloat)1e24, weight / max((tfloat)1e-6, update));
	else
		d_newweight[id] = weight;
}

__global__ void CorrectGriddingKernel(tfloat* d_volume, uint dim)
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= dim)
		return;
	uint idy = blockIdx.y * blockDim.y + threadIdx.y;
	if (idy >= dim)
		return;
	uint idz = blockIdx.z * blockDim.z + threadIdx.z;
	if (idz >= dim)
		return;

	int x = (int)idx - (int)(dim / 2);
	int y = (int)idy - (int)(dim / 2);
	int z = (int)idz - (int)(dim / 2);

	tfloat r = sqrt((tfloat)(x * x + y * y + z * z));
	r /= (tfloat)dim;

	if (r > 0)
		d_volume[(idz * dim + idy) * dim + idx] /= sinc(r) * sinc(r);
}