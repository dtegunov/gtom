#include "Prerequisites.cuh"
#include "IO.cuh"

void ReadEM(string path, void** data, EM_DATATYPE datatype, int nframe)
{
	ifstream inputfile(path, ios::in|ios::binary|ios::ate);
	inputfile.seekg(0, ios::beg);

	HeaderEM header = ReadEMHeader(inputfile);

	size_t datasize;
	if (nframe >= 0)
		datasize = Elements2(header.dimensions) * EM_DATATYPE_SIZE[(int)header.mode];
	else
		datasize = Elements(header.dimensions) * EM_DATATYPE_SIZE[(int)header.mode];

	cudaMallocHost(data, datasize);

	if(nframe >= 0)
		inputfile.seekg(datasize * nframe, ios::cur);

	inputfile.read((char*)*data, datasize);

	inputfile.close();
}

HeaderEM ReadEMHeader(string path)
{
	ifstream inputfile(path, ios::in | ios::binary);
	inputfile.seekg(0, ios::beg);

	HeaderEM header = ReadEMHeader(inputfile);
	inputfile.close();

	return header;
}

HeaderEM ReadEMHeader(std::ifstream &inputfile)
{
	HeaderEM header;
	char* headerp = (char*)&header;

	inputfile.read(headerp, sizeof(HeaderEM));

	return header;
}