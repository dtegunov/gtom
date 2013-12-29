#include "../Prerequisites.cuh"
#include "../Functions.cuh"

void ReadEM(string path, void** data, EM_DATATYPE &datatype, int nframe)
{
	char* header = (char*)malloc(512 * sizeof(char));

	ifstream inputfile(path, ios::in|ios::binary|ios::ate);
	inputfile.seekg(0, ios::beg);
	inputfile.read(header, 512 * sizeof(char));
	
	int3 dims;
	dims.x = ((int*)header)[1];
	dims.y = ((int*)header)[2];
	dims.z = 1;
	datatype = (EM_DATATYPE)header[3];

	free(header);

	size_t bytesperfield = 1;
	if(datatype == EM_DATATYPE::EM_SHORT)
		bytesperfield = 2;
	else if(datatype == EM_DATATYPE::EM_LONG || datatype == EM_DATATYPE::EM_SINGLE)
		bytesperfield = 4;
	else if(datatype == EM_DATATYPE::EM_SINGLECOMPLEX || datatype == EM_DATATYPE::EM_DOUBLE)
		bytesperfield = 8;
	else if(datatype == EM_DATATYPE::EM_DOUBLECOMPLEX)
		bytesperfield = 16;

	size_t datasize = Elements(dims) * bytesperfield;
	cudaMallocHost(data, datasize);

	inputfile.seekg(datasize * nframe, ios::cur);

	inputfile.read((char*)*data, datasize);

	inputfile.close();
}

void ReadEMDims(string path, int3 &dims)
{
	char* header = (char*)malloc(1024 * sizeof(char));

	ifstream inputfile(path, ios::in|ios::binary);
	inputfile.seekg(0, ios::beg);
	inputfile.read(header, 1024 * sizeof(char));

	dims.x = ((int*)header)[1];
	dims.y = ((int*)header)[2];
	dims.z = ((int*)header)[3];

	free(header);
}