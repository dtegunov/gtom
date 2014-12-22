#include "Prerequisites.cuh"
#include "IO.cuh"

void ReadMRC(string path, void** data, MRC_DATATYPE datatype, int nframe)
{

	ifstream inputfile(path, ios::in|ios::binary);
	inputfile.seekg(0, ios::beg);

	HeaderMRC header = ReadMRCHeader(inputfile);
	
	size_t datasize;
	if (nframe >= 0)
		datasize = Elements2(header.dimensions) * MRC_DATATYPE_SIZE[(int)header.mode];
	else
		datasize = Elements(header.dimensions) * MRC_DATATYPE_SIZE[(int)header.mode];

	cudaMallocHost(data, datasize);

	if (nframe >= 0)
		inputfile.seekg(datasize * nframe, ios::cur);

	inputfile.read((char*)*data, datasize);

	inputfile.close();
}

HeaderMRC ReadMRCHeader(string path)
{
	ifstream inputfile(path, ios::in|ios::binary);
	inputfile.seekg(0, ios::beg);

	HeaderMRC header = ReadMRCHeader(inputfile);
	inputfile.close();

	return header;
}

HeaderMRC ReadMRCHeader(std::ifstream &inputfile)
{
	HeaderMRC header;
	char* headerp = (char*)&header;

	inputfile.read(headerp, sizeof(HeaderMRC));
	inputfile.seekg(header.extendedbytes, ios::cur);

	return header;
}