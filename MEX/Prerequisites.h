#include "mex.h"
#include "gpu/mxGPUArray.h"
#include "../Functions.cuh"
#include "../Prerequisites.cuh"

size_t RealToFTComplexDims(int const ndims, mwSize const* dims);
size_t RealToFTComplexDims(int const ndims, int3 const dims);
int3 MWDimsToInt3(int const ndims, mwSize const* dims);
int3 MWDimsToInt3(int const ndims, double* dims);

class mxArrayAdapter
{
	void* createdcopy;
	void* d_createdcopy;

public:
	mxArray* underlyingarray;
	mxArrayAdapter(mxArray const*);
	~mxArrayAdapter();
	tfloat* GetAsManagedTFloat();
	tfloat* GetAsManagedDeviceTFloat();
	tcomplex* GetAsManagedTComplex();
	tcomplex* GetAsManagedDeviceTComplex();
	tfloat* GetAsUnmanagedTFloat();
	tfloat* GetAsUnmanagedDeviceTFloat();
	tcomplex* GetAsUnmanagedTComplex();
	tcomplex* GetAsUnmanagedDeviceTComplex();
	void SetFromTFloat(tfloat*);
	void SetFromTComplex(tcomplex*);
	void SetFromDeviceTFloat(tfloat*);
	void SetFromDeviceTComplex(tcomplex*);
};