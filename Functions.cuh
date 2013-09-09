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

//Fourier transform

extern "C" __declspec(dllexport) void __stdcall FFT(tfloat* const d_input, tcomplex* const d_output, int const ndimensions, int3 const dimensions);