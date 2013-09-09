#pragma once
#include "cufft.h"

//#define TOM_DOUBLE

#ifdef TOM_DOUBLE
	typedef double tfloat;
	typedef cufftDoubleComplex tcomplex;
#else
	typedef float tfloat;
	typedef cufftComplex tcomplex;
#endif

//Type conversion
extern "C" __declspec(dllexport) tfloat* __stdcall ConvertDoubleToTFloat(double const* original, size_t const n);
extern "C" __declspec(dllexport) double* __stdcall ConvertTFloatToDouble(float const* original, size_t const n);
extern "C" __declspec(dllexport) tfloat* __stdcall ConvertMWComplexToTFloatInterleaved(double const* original, size_t const n);
extern "C" __declspec(dllexport) double* __stdcall ConvertSingleInterleavedToMWComplex(tfloat const* original, size_t const n);

//Fourier transform

extern "C" __declspec(dllexport) void __stdcall FFT(tfloat* const d_input, tcomplex* const d_output, int const ndimensions, int3 const dimensions);