#include "Prerequisites.cuh"
#include "Helper.cuh"
#include "IO.cuh"

void ReadMRC(string path, void** data, int nframe)
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

void WriteMRC(void* data, HeaderMRC header, string path)
{
	FILE* outputfile = fopen(path.c_str(), "wb");
	_fseeki64(outputfile, 0L, SEEK_SET);

	fwrite(&header, sizeof(HeaderMRC), 1, outputfile);
	
	size_t elementsize = MRC_DATATYPE_SIZE[(int)header.mode];
	fwrite(data, elementsize, Elements(header.dimensions), outputfile);

	fclose(outputfile);
}

void WriteMRC(tfloat* data, int3 dims, string path)
{
	HeaderMRC header;
	header.dimensions = dims;
#ifdef GTOM_DOUBLE
	throw;	// MRC can't do double!
#else
	header.mode = MRC_FLOAT;
#endif

	float minval = 1e30f, maxval = -1e30f;
	size_t elements = Elements(dims);
	for (size_t i = 0; i < elements; i++)
	{
		minval = min(minval, data[i]);
		maxval = max(maxval, data[i]);
	}
	header.maxvalue = maxval;
	header.minvalue = minval;

	WriteMRC(data, header, path);
}

void d_WriteMRC(tfloat* d_data, int3 dims, string path)
{
	tfloat* h_data = (tfloat*)MallocFromDeviceArray(d_data, Elements(dims) * sizeof(tfloat));

	WriteMRC(h_data, dims, path);

	free(h_data);
}