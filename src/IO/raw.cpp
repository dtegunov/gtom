#include "Prerequisites.cuh"


void ReadRAW(string path, void** data, EM_DATATYPE datatype, int3 dims, int nframe, size_t headerbytes)
{
	ifstream inputfile(path, ios::in|ios::binary|ios::ate);
	inputfile.seekg(0, ios::beg);

	size_t bytesperfield = 1;
	if(datatype == EM_DATATYPE::EM_SHORT)
		bytesperfield = 2;
	else if(datatype == EM_DATATYPE::EM_LONG || datatype == EM_DATATYPE::EM_SINGLE)
		bytesperfield = 4;
	else if(datatype == EM_DATATYPE::EM_SINGLECOMPLEX || datatype == EM_DATATYPE::EM_DOUBLE)
		bytesperfield = 8;
	else if(datatype == EM_DATATYPE::EM_DOUBLECOMPLEX)
		bytesperfield = 16;

	size_t datasize = Elements(toInt3(dims.x, dims.y, 1)) * bytesperfield;
	cudaMallocHost(data, datasize);

	inputfile.seekg(headerbytes + datasize * nframe, ios::cur);

	inputfile.read((char*)*data, datasize);

	inputfile.close();
}