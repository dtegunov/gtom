#include "Prerequisites.h"

size_t RealToFTComplexDims(int const ndims, mwSize const* dims)
{
	size_t elements = dims[0] / 2 + 1;
	for(int i = 1; i < ndims; i++)
		elements *= dims[i];

	return elements;
}

size_t RealToFTComplexDims(int const ndims, int3 const dims)
{
	size_t elements = dims.x / 2 + 1;
	if(ndims > 1)
		elements *= dims.y;
	if(ndims > 2)
		elements *= dims.z;

	return elements;
}

int3 MWDimsToInt3(int const ndims, mwSize const* dims)
{
	int3 val;
	val.x = dims[0];
	if(ndims > 1)
		val.y = dims[1];
	else
		val.y = 1;
	if(ndims > 2)
		val.z = dims[2];
	else
		val.z = 1;

	return val;
}

int3 MWDimsToInt3(int const ndims, double* dims)
{
	int3 val;
	val.x = dims[0];
	if(ndims > 1)
		val.y = dims[1];
	else
		val.y = 1;
	if(ndims > 2)
		val.z = dims[2];
	else
		val.z = 1;

	return val;
}


//////////////////
//mxArrayAdapter//
//////////////////

mxArrayAdapter::mxArrayAdapter(mxArray const* underlying)
{
	underlyingarray = (mxArray*)underlying;
	createdcopy = NULL;
	d_createdcopy = NULL;
}

mxArrayAdapter::~mxArrayAdapter()
{
	if(createdcopy != NULL)
		free(createdcopy);

	if(d_createdcopy != NULL)
		cudaFree(d_createdcopy);
}

tfloat* mxArrayAdapter::GetAsManagedTFloat()
{
	createdcopy = this->GetAsUnmanagedTFloat();
	return (tfloat*)createdcopy;
}

tfloat* mxArrayAdapter::GetAsManagedDeviceTFloat()
{
	d_createdcopy = this->GetAsUnmanagedDeviceTFloat();
	return (tfloat*)d_createdcopy;
}

tcomplex* mxArrayAdapter::GetAsManagedTComplex()
{
	createdcopy = this->GetAsUnmanagedTComplex();
	return (tcomplex*)createdcopy;
}

tcomplex* mxArrayAdapter::GetAsManagedDeviceTComplex()
{
	d_createdcopy = this->GetAsUnmanagedDeviceTComplex();
	return (tcomplex*)d_createdcopy;
}

tfloat* mxArrayAdapter::GetAsUnmanagedTFloat()
{
	if (mxGetClassID(underlyingarray) == mxDOUBLE_CLASS)
	{
		if (!IS_TFLOAT_DOUBLE)
			return ConvertToTFloat<double>((double*)mxGetPr(underlyingarray), mxGetNumberOfElements(underlyingarray));
		else
		{
			tfloat* h_copy = (tfloat*)malloc(mxGetNumberOfElements(underlyingarray) * sizeof(tfloat));
			memcpy(h_copy, (tfloat*)mxGetPr(underlyingarray), mxGetNumberOfElements(underlyingarray) * sizeof(tfloat));
			return h_copy;
		}
	}
	else if (mxGetClassID(underlyingarray) == mxSINGLE_CLASS)
	{
		if (IS_TFLOAT_DOUBLE)
			return ConvertToTFloat<float>((float*)mxGetPr(underlyingarray), mxGetNumberOfElements(underlyingarray));
		else
		{
			tfloat* h_copy = (tfloat*)malloc(mxGetNumberOfElements(underlyingarray) * sizeof(tfloat));
			memcpy(h_copy, (tfloat*)mxGetPr(underlyingarray), mxGetNumberOfElements(underlyingarray) * sizeof(tfloat));
			return h_copy;
		}
	}
	else if (mxGetClassID(underlyingarray) == mxINT32_CLASS)
		return ConvertToTFloat<int>((int*)mxGetPr(underlyingarray), mxGetNumberOfElements(underlyingarray));
	else if (mxGetClassID(underlyingarray) == mxINT16_CLASS)
		return ConvertToTFloat<short>((short*)mxGetPr(underlyingarray), mxGetNumberOfElements(underlyingarray));
	else if (mxGetClassID(underlyingarray) == mxUINT32_CLASS)
		return ConvertToTFloat<uint>((uint*)mxGetPr(underlyingarray), mxGetNumberOfElements(underlyingarray));
	else if (mxGetClassID(underlyingarray) == mxUINT16_CLASS)
		return ConvertToTFloat<ushort>((ushort*)mxGetPr(underlyingarray), mxGetNumberOfElements(underlyingarray));
	else
		throw;
}

tfloat* mxArrayAdapter::GetAsUnmanagedDeviceTFloat()
{
	tfloat* d_copy;

	if(mxGetClassID(underlyingarray) == mxDOUBLE_CLASS)
	{
		double* d_original = (double*)CudaMallocFromHostArray(mxGetPr(underlyingarray), mxGetNumberOfElements(underlyingarray) * sizeof(double));
		if(!IS_TFLOAT_DOUBLE)
		{
			cudaMalloc((void**)&d_copy, mxGetNumberOfElements(underlyingarray) * sizeof(tfloat));
			d_ConvertToTFloat<double>(d_original, d_copy, mxGetNumberOfElements(underlyingarray));
			cudaFree(d_original);
		}
		else
			d_copy = (tfloat*)d_original;
	}
	else if (mxGetClassID(underlyingarray) == mxSINGLE_CLASS)
	{
		float* d_original = (float*)CudaMallocFromHostArray(mxGetPr(underlyingarray), mxGetNumberOfElements(underlyingarray) * sizeof(float));
		if (IS_TFLOAT_DOUBLE)
		{
			cudaMalloc((void**)&d_copy, mxGetNumberOfElements(underlyingarray) * sizeof(tfloat));
			d_ConvertToTFloat<float>(d_original, d_copy, mxGetNumberOfElements(underlyingarray));
			cudaFree(d_original);
		}
		else
			d_copy = (tfloat*)d_original;
	}
	else if (mxGetClassID(underlyingarray) == mxINT32_CLASS)
	{
		int* d_original = (int*)CudaMallocFromHostArray(mxGetPr(underlyingarray), mxGetNumberOfElements(underlyingarray) * sizeof(int));
		cudaMalloc((void**)&d_copy, mxGetNumberOfElements(underlyingarray) * sizeof(tfloat));
		d_ConvertToTFloat<int>(d_original, d_copy, mxGetNumberOfElements(underlyingarray));
		cudaFree(d_original);
	}
	else if (mxGetClassID(underlyingarray) == mxUINT32_CLASS)
	{
		uint* d_original = (uint*)CudaMallocFromHostArray(mxGetPr(underlyingarray), mxGetNumberOfElements(underlyingarray) * sizeof(uint));
		cudaMalloc((void**)&d_copy, mxGetNumberOfElements(underlyingarray) * sizeof(tfloat));
		d_ConvertToTFloat<uint>(d_original, d_copy, mxGetNumberOfElements(underlyingarray));
		cudaFree(d_original);
	}
	else if (mxGetClassID(underlyingarray) == mxINT16_CLASS)
	{
		short* d_original = (short*)CudaMallocFromHostArray(mxGetPr(underlyingarray), mxGetNumberOfElements(underlyingarray) * sizeof(short));
		cudaMalloc((void**)&d_copy, mxGetNumberOfElements(underlyingarray) * sizeof(tfloat));
		d_ConvertToTFloat<short>(d_original, d_copy, mxGetNumberOfElements(underlyingarray));
		cudaFree(d_original);
	}
	else if (mxGetClassID(underlyingarray) == mxINT16_CLASS)
	{
		ushort* d_original = (ushort*)CudaMallocFromHostArray(mxGetPr(underlyingarray), mxGetNumberOfElements(underlyingarray) * sizeof(ushort));
		cudaMalloc((void**)&d_copy, mxGetNumberOfElements(underlyingarray) * sizeof(tfloat));
		d_ConvertToTFloat<ushort>(d_original, d_copy, mxGetNumberOfElements(underlyingarray));
		cudaFree(d_original);
	}
	else
		throw;
	
	return d_copy;
}

tcomplex* mxArrayAdapter::GetAsUnmanagedTComplex()
{
	if(mxGetClassID(underlyingarray) == mxDOUBLE_CLASS && !IS_TFLOAT_DOUBLE)
		return ConvertSplitComplexToTComplex<double>((double*)mxGetPr(underlyingarray),
													 (double*)mxGetPi(underlyingarray),
													 mxGetNumberOfElements(underlyingarray));
	else
		return ConvertSplitComplexToTComplex<float>((float*)mxGetPr(underlyingarray), 
													 (float*)mxGetPi(underlyingarray),
													 mxGetNumberOfElements(underlyingarray));
}

tcomplex* mxArrayAdapter::GetAsUnmanagedDeviceTComplex()
{
	tcomplex* d_copy;
	cudaMalloc((void**)&d_copy, mxGetNumberOfElements(underlyingarray) * sizeof(tcomplex));

	if(mxGetClassID(underlyingarray) == mxDOUBLE_CLASS)
	{
		double* d_originalr = (double*)CudaMallocFromHostArray(mxGetPr(underlyingarray), mxGetNumberOfElements(underlyingarray) * sizeof(double));
		double* d_originali = (double*)CudaMallocFromHostArray(mxGetPi(underlyingarray), mxGetNumberOfElements(underlyingarray) * sizeof(double));
		d_ConvertSplitComplexToTComplex<double>(d_originalr, d_originali, d_copy, mxGetNumberOfElements(underlyingarray));
		cudaFree(d_originalr);
		cudaFree(d_originali);
	}
	else
	{
		float* d_originalr = (float*)CudaMallocFromHostArray(mxGetPr(underlyingarray), mxGetNumberOfElements(underlyingarray) * sizeof(float));
		float* d_originali = (float*)CudaMallocFromHostArray(mxGetPi(underlyingarray), mxGetNumberOfElements(underlyingarray) * sizeof(float));
		d_ConvertSplitComplexToTComplex<float>(d_originalr, d_originali, d_copy, mxGetNumberOfElements(underlyingarray));
		cudaFree(d_originalr);
		cudaFree(d_originali);
	}
	
	cudaDeviceSynchronize();
	return d_copy;
}

void mxArrayAdapter::SetFromTFloat(tfloat* original)
{
	if((mxGetClassID(underlyingarray) == mxDOUBLE_CLASS) == (!IS_TFLOAT_DOUBLE))
	{
		if(mxGetClassID(underlyingarray) == mxDOUBLE_CLASS)
			ConvertTFloatTo<double>(original, (double*)mxGetPr(underlyingarray), mxGetNumberOfElements(underlyingarray));
		else if(mxGetClassID(underlyingarray) == mxSINGLE_CLASS)
			ConvertTFloatTo<float>(original, (float*)mxGetPr(underlyingarray), mxGetNumberOfElements(underlyingarray));
	}
	else
		memcpy(mxGetPr(underlyingarray), original, mxGetNumberOfElements(underlyingarray) * (mxGetClassID(underlyingarray) == mxDOUBLE_CLASS ? sizeof(double) : sizeof(float)));
}

void mxArrayAdapter::SetFromTComplex(tcomplex* original)
{
	if(mxGetClassID(underlyingarray) == mxDOUBLE_CLASS)
		ConvertTComplexToSplitComplex(original, (double*)mxGetPr(underlyingarray), (double*)mxGetPi(underlyingarray),  mxGetNumberOfElements(underlyingarray));
	else if(mxGetClassID(underlyingarray) == mxSINGLE_CLASS)
		ConvertTComplexToSplitComplex(original, (float*)mxGetPr(underlyingarray), (float*)mxGetPi(underlyingarray), mxGetNumberOfElements(underlyingarray));
}

void mxArrayAdapter::SetFromDeviceTFloat(tfloat* d_original)
{
	void* d_copy;

	if((mxGetClassID(underlyingarray) == mxDOUBLE_CLASS))
	{
		if(!IS_TFLOAT_DOUBLE)
		{
			cudaMalloc((void**)&d_copy, mxGetNumberOfElements(underlyingarray) * sizeof(double));
			d_ConvertTFloatTo<double>(d_original, (double*)d_copy, mxGetNumberOfElements(underlyingarray));
		}
		else
			d_copy = d_original;

		cudaMemcpy(mxGetPr(underlyingarray), d_copy, mxGetNumberOfElements(underlyingarray) * sizeof(double), cudaMemcpyDeviceToHost);
	}
	else
	{
		if(IS_TFLOAT_DOUBLE)
		{
			cudaMalloc((void**)&d_copy, mxGetNumberOfElements(underlyingarray) * sizeof(float));
			d_ConvertTFloatTo<float>(d_original, (float*)d_copy, mxGetNumberOfElements(underlyingarray));
		}
		else
			d_copy = d_original;

		cudaMemcpy(mxGetPr(underlyingarray), d_copy, mxGetNumberOfElements(underlyingarray) * sizeof(float), cudaMemcpyDeviceToHost);
	}
	
	if(d_copy != d_original)
		cudaFree(d_copy);
}

void mxArrayAdapter::SetFromDeviceTComplex(tcomplex* d_original)
{
	void* d_copyr;
	void* d_copyi;
	
	if((mxGetClassID(underlyingarray) == mxDOUBLE_CLASS))
	{
		cudaMalloc((void**)&d_copyr, mxGetNumberOfElements(underlyingarray) * sizeof(double));
		cudaMalloc((void**)&d_copyi, mxGetNumberOfElements(underlyingarray) * sizeof(double));
		d_ConvertTComplexToSplitComplex<double>(d_original, (double*)d_copyr, (double*)d_copyi, mxGetNumberOfElements(underlyingarray));

		cudaMemcpy(mxGetPr(underlyingarray), d_copyr, mxGetNumberOfElements(underlyingarray) * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(mxGetPi(underlyingarray), d_copyi, mxGetNumberOfElements(underlyingarray) * sizeof(double), cudaMemcpyDeviceToHost);
	}
	else
	{
		cudaMalloc((void**)&d_copyr, mxGetNumberOfElements(underlyingarray) * sizeof(float));
		cudaMalloc((void**)&d_copyi, mxGetNumberOfElements(underlyingarray) * sizeof(float));
		d_ConvertTComplexToSplitComplex<float>(d_original, (float*)d_copyr, (float*)d_copyi, mxGetNumberOfElements(underlyingarray));

		cudaMemcpy(mxGetPr(underlyingarray), d_copyr, mxGetNumberOfElements(underlyingarray) * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(mxGetPi(underlyingarray), d_copyi, mxGetNumberOfElements(underlyingarray) * sizeof(float), cudaMemcpyDeviceToHost);
	}

	cudaDeviceSynchronize();
	cudaFree(d_copyr);
	cudaFree(d_copyi);
}