#include "Prerequisites.cuh"
#include "IO.cuh"


void ReadRAW(string path, void** data, EM_DATATYPE datatype, int3 dims, size_t headerbytes, int nframe)
{
	ifstream inputfile(path, ios::in|ios::binary|ios::ate);
	inputfile.seekg(0, ios::beg);

	size_t bytesperfield = EM_DATATYPE_SIZE[(int)datatype];

	size_t datasize = Elements(dims) * bytesperfield;
	cudaMallocHost(data, datasize);

	if (nframe >= 0)
		inputfile.seekg(headerbytes + datasize * nframe, ios::cur);

	inputfile.read((char*)*data, datasize);

	inputfile.close();
}