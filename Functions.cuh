#pragma once
#include "cufft.h"

//Fourier transform

extern "C" __declspec(dllexport) void __stdcall FFT(cufftReal* const d_input, cufftComplex* const d_output, int const ndimensions, int3 const dimensions);