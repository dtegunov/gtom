#include "Prerequisites.cuh"
#include "IO.cuh"


void ReadRAW(string path, void** data, EM_DATATYPE datatype, int3 dims, size_t headerbytes, int nframe)
{
	FILE* inputfile = fopen(path.c_str(), "rb");
	_fseeki64(inputfile, 0L, SEEK_SET);

	size_t bytesperfield = EM_DATATYPE_SIZE[(int)datatype];

	size_t datasize = Elements(dims) * bytesperfield;
	cudaMallocHost(data, datasize);

	if (nframe >= 0)
		_fseeki64(inputfile, headerbytes + datasize * (size_t)nframe, SEEK_CUR);

	fread(*data, sizeof(char), datasize, inputfile);

	fclose(inputfile);
}