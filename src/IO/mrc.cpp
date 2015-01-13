#include "Prerequisites.cuh"
#include "IO.cuh"

void ReadMRC(string path, void** data, MRC_DATATYPE datatype, int nframe)
{
	FILE* inputfile = fopen(path.c_str(), "rb");
	_fseeki64(inputfile, 0L, SEEK_SET);

	HeaderMRC header = ReadMRCHeader(inputfile);
	
	size_t datasize;
	if (nframe >= 0)
		datasize = Elements2(header.dimensions) * MRC_DATATYPE_SIZE[(int)header.mode];
	else
		datasize = Elements(header.dimensions) * MRC_DATATYPE_SIZE[(int)header.mode];

	cudaMallocHost(data, datasize);

	if (nframe >= 0)
		_fseeki64(inputfile, datasize * (size_t)nframe, SEEK_CUR);

	fread(*data, sizeof(char), datasize, inputfile);

	fclose(inputfile);
}

HeaderMRC ReadMRCHeader(string path)
{
	FILE* inputfile = fopen(path.c_str(), "rb");
	_fseeki64(inputfile, 0L, SEEK_SET);

	HeaderMRC header = ReadMRCHeader(inputfile);
	fclose(inputfile);

	return header;
}

HeaderMRC ReadMRCHeader(FILE* inputfile)
{
	HeaderMRC header;
	char* headerp = (char*)&header;

	fread(headerp, sizeof(char), sizeof(HeaderMRC), inputfile);
	_fseeki64(inputfile, (long)header.extendedbytes, SEEK_CUR);

	return header;
}