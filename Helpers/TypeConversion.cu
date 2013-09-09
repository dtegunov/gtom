#include "..\Prerequisites.cuh"
#include "..\Functions.cuh"

#ifndef BlockSize
	#define BlockSize 1024
#endif

__declspec(dllexport) tfloat* __stdcall ConvertDoubleToTFloat(double const* original, size_t const n)
{
	tfloat* converted = (tfloat*)malloc(n * sizeof(tfloat));

	#ifdef TOM_DOUBLE
		memcpy(converted, original, n * sizeof(tfloat));
	#else
		#pragma omp for schedule(dynamic, BlockSize)
		for(int i = 0; i < n; i++)
			converted[i] = (tfloat)original[i];
	#endif

	return converted;
}

__declspec(dllexport) double* __stdcall ConvertTFloatToDouble(float const* original, size_t const n)
{
	double* converted = (double*)malloc(n * sizeof(double));

	#ifdef TOM_DOUBLE
		memcpy(converted, original, n * sizeof(tfloat));
	#else
		#pragma omp for schedule(dynamic, BlockSize)
		for(int i = 0; i < n; i++)
			converted[i] = (double)original[i];
	#endif

	return converted;
}

__declspec(dllexport) tfloat* __stdcall ConvertMWComplexToTFloatInterleaved(double const* original, size_t const n)
{
	tfloat* converted = (tfloat*)malloc(n * 2 * sizeof(tfloat));

	#pragma omp for schedule(dynamic, BlockSize)
	for(int i = 0; i < n; i++)
		converted[i * 2] = (tfloat)original[i];

	#pragma omp for schedule(dynamic, BlockSize)
	for(int i = 0; i < n; i++)
		converted[i * 2 + 1] = (tfloat)original[n + i];

	return converted;
}

__declspec(dllexport) double* __stdcall ConvertSingleInterleavedToMWComplex(tfloat const* original, size_t const n)
{
	double* converted = (double*)malloc(n * 2 * sizeof(double));

		#pragma omp for schedule(dynamic, BlockSize)
		for(int i = 0; i < n; i++)
		{
			converted[i] = (double)original[i * 2];
			converted[n + i] = (double)original[i * 2 + 1];
		}

	return converted;
}