#include "Prerequisites.cuh"
#include "IO.cuh"

void ReadEM(string path, void** data, int nframe)
{
	FILE* inputfile = fopen(path.c_str(), "rb");
	_fseeki64(inputfile, 0L, SEEK_SET);

	HeaderEM header = ReadEMHeader(inputfile);

	size_t datasize;
	if (nframe >= 0)
		datasize = Elements2(header.dimensions) * EM_DATATYPE_SIZE[(int)header.mode];
	else
		datasize = Elements(header.dimensions) * EM_DATATYPE_SIZE[(int)header.mode];

	cudaMallocHost(data, datasize);

	if (nframe >= 0)
		_fseeki64(inputfile, datasize * (size_t)nframe, SEEK_CUR);

	fread(*data, sizeof(char), datasize, inputfile);

	fclose(inputfile);
}

HeaderEM ReadEMHeader(string path)
{
	FILE* inputfile = fopen(path.c_str(), "rb");
	_fseeki64(inputfile, 0L, SEEK_SET);

	HeaderEM header = ReadEMHeader(inputfile);
	fclose(inputfile);

	return header;
}

HeaderEM ReadEMHeader(FILE* inputfile)
{
	HeaderEM header;
	char* headerp = (char*)&header;

	fread(headerp, sizeof(char), sizeof(HeaderEM), inputfile);

	return header;
}