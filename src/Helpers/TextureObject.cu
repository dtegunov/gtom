#include "Prerequisites.cuh"

void d_BindTextureToArray(tfloat* d_input, cudaArray* &d_createdarray, cudaTextureObject_t &texture, int2 dims, cudaTextureFilterMode filtermode, bool normalizedcoords)
{
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<tfloat>();
	cudaArray* d_inputArray;
	cudaMallocArray(&d_inputArray, &desc, dims.x, dims.y);
	cudaMemcpyToArray(d_inputArray, 0, 0, d_input, dims.x * dims.y * sizeof(tfloat), cudaMemcpyDeviceToDevice);

	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = d_inputArray;

	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.filterMode = filtermode;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = normalizedcoords;

	cudaTextureObject_t texObj = 0;
	cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

	d_createdarray = d_inputArray;
	texture = texObj;
}