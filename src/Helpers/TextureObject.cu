#include "Prerequisites.cuh"
#include "Helper.cuh"

void d_BindTextureToArray(tfloat* d_input, cudaArray* &d_createdarray, cudaTextureObject_t &texture, int2 dims, cudaTextureFilterMode filtermode, bool normalizedcoords)
{
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<tfloat>();
	cudaArray* a_input;
	cudaMallocArray(&a_input, &desc, dims.x, dims.y);
	cudaMemcpyToArray(a_input, 0, 0, d_input, dims.x * dims.y * sizeof(tfloat), cudaMemcpyDeviceToDevice);

	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = a_input;

	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.filterMode = filtermode;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = normalizedcoords;
	texDesc.addressMode[0] = cudaAddressModeWrap;
	texDesc.addressMode[1] = cudaAddressModeWrap;
	texDesc.addressMode[2] = cudaAddressModeWrap;
	cudaTextureObject_t texObj = 0;
	cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

	d_createdarray = a_input;
	texture = texObj;
}

void d_BindTextureTo3DArray(tfloat* d_input, cudaArray* &d_createdarray, cudaTextureObject_t &texture, int3 dims, cudaTextureFilterMode filtermode, bool normalizedcoords)
{
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<tfloat>();
	cudaExtent extent = make_cudaExtent(dims.x, dims.y, dims.z);
	cudaArray* a_input;
	cudaMalloc3DArray(&a_input, &desc, extent);

	cudaPitchedPtr p_input = CopyVolumeDeviceToDevice(d_input, dims);

	cudaMemcpy3DParms p = { 0 };
	p.extent = extent;
	p.srcPtr = p_input;
	p.dstArray = a_input;
	p.kind = cudaMemcpyDeviceToDevice;
	cudaMemcpy3D(&p);

	cudaFree(p_input.ptr);

	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = a_input;

	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.filterMode = filtermode;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = normalizedcoords;
	texDesc.addressMode[0] = cudaAddressModeWrap;
	texDesc.addressMode[1] = cudaAddressModeWrap;
	texDesc.addressMode[2] = cudaAddressModeWrap;
	cudaTextureObject_t texObj = 0;
	cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

	d_createdarray = a_input;
	texture = texObj;
}