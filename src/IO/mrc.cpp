#include "Prerequisites.cuh"

void ReadMRC(string path, void** data, EM_DATATYPE &datatype, int nframe, bool flipx)
{
	char* header = (char*)malloc(1024 * sizeof(char));

	ifstream inputfile(path, ios::in|ios::binary);
	inputfile.seekg(0, ios::beg);
	//#pragma omp critical
	inputfile.read(header, 1024 * sizeof(char));

	int mode = ((int*)header)[3];
	if(mode == 0)
		datatype = EM_DATATYPE::EM_BYTE;
	else if(mode == 1 || mode == 6)
		datatype = EM_DATATYPE::EM_SHORT;
	else if(mode == 2)
		datatype = EM_DATATYPE::EM_SINGLE;
	else if(mode == 3)
		datatype = EM_DATATYPE::EM_SHORTCOMPLEX;
	else if(mode == 4)
		datatype = EM_DATATYPE::EM_SINGLECOMPLEX;

	size_t bytesperfield = 1;
	if(datatype == EM_DATATYPE::EM_SHORT)
		bytesperfield = 2;
	else if(datatype == EM_DATATYPE::EM_SHORTCOMPLEX || datatype == EM_DATATYPE::EM_LONG || datatype == EM_DATATYPE::EM_SINGLE)
		bytesperfield = 4;
	else if(datatype == EM_DATATYPE::EM_SINGLECOMPLEX || datatype == EM_DATATYPE::EM_DOUBLE)
		bytesperfield = 8;
	else if(datatype == EM_DATATYPE::EM_DOUBLECOMPLEX)
		bytesperfield = 16;

	int extendedsize = header[23];
	void* extendedheader = malloc(extendedsize * sizeof(char));
	inputfile.read((char*)extendedheader, extendedsize * sizeof(char));
	
	int3 dims;
	dims.x = ((int*)header)[0];
	dims.y = ((int*)header)[1];
	dims.z = 1;
	size_t datasize = Elements(dims) * bytesperfield;
	cudaMallocHost(data, datasize);

	inputfile.seekg(datasize * nframe, ios::cur);

	inputfile.read((char*)*data, datasize);

	inputfile.close();
	free(header);
	free(extendedheader);

	if(!flipx)
		return;

	size_t layersize = dims.x * dims.y;
	size_t linewidth = dims.x;
	void* flipbuffer = malloc(linewidth * bytesperfield);
	size_t offsetlayer, offsetrow;
	int dimsxminusone = dims.x - 1;

	for(int z = 0; z < dims.z; z++)
	{
		offsetlayer = z * layersize;
		if(datatype == EM_DATATYPE::EM_BYTE)
			for (int y = 0; y < dims.y; y++)
			{
				offsetrow = offsetlayer + y * linewidth;
				memcpy(flipbuffer, ((char*)*data) + offsetrow, linewidth);
				for (int x = 0; x < dims.x; x++)
					((char*)*data)[offsetrow + x] = ((char*)flipbuffer)[dimsxminusone - x];
			}		
		else if(datatype == EM_DATATYPE::EM_SHORT)
			for (int y = 0; y < dims.y; y++)
			{
				offsetrow = offsetlayer + y * linewidth;
				memcpy(flipbuffer, ((short*)*data) + offsetrow, linewidth);
				for (int x = 0; x < dims.x; x++)
					((short*)*data)[offsetrow + x] = ((short*)flipbuffer)[dimsxminusone - x];
			}		
		else if(datatype == EM_DATATYPE::EM_SHORTCOMPLEX || datatype == EM_DATATYPE::EM_LONG || datatype == EM_DATATYPE::EM_SINGLE)
			for (int y = 0; y < dims.y; y++)
			{
				offsetrow = offsetlayer + y * linewidth;
				memcpy(flipbuffer, ((float*)*data) + offsetrow, linewidth);
				for (int x = 0; x < dims.x; x++)
					((float*)*data)[offsetrow + x] = ((float*)flipbuffer)[dimsxminusone - x];
			}		
		else if(datatype == EM_DATATYPE::EM_SINGLECOMPLEX || datatype == EM_DATATYPE::EM_DOUBLE)
			for (int y = 0; y < dims.y; y++)
			{
				offsetrow = offsetlayer + y * linewidth;
				memcpy(flipbuffer, ((double*)*data) + offsetrow, linewidth);
				for (int x = 0; x < dims.x; x++)
					((double*)*data)[offsetrow + x] = ((double*)flipbuffer)[dimsxminusone - x];
			}		
	}

	free(flipbuffer);
}

void ReadMRCDims(string path, int3 &dims)
{
	char* header = (char*)malloc(1024 * sizeof(char));

	ifstream inputfile(path, ios::in|ios::binary);
	inputfile.seekg(0, ios::beg);
	inputfile.read(header, 1024 * sizeof(char));

	dims.x = ((int*)header)[0];
	dims.y = ((int*)header)[1];
	dims.z = ((int*)header)[2];

	free(header);
}