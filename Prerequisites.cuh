#ifndef GTOM_PREREQUISITES_H
#define GTOM_PREREQUISITES_H

#include "CubicSplines/internal/cutil_math_bugfixes.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <cufft.h>
#include <cublas_v2.h>
#include <omp.h>
#include <iostream>
#include <sstream>
#include <fstream>

//#define GLM_FORCE_RADIANS
//#include "glm\glm.hpp"
//#include "glm\gtc\matrix_transform.hpp"
//#include "glm\gtx\quaternion.hpp"
//#include "glm\gtx\euler_angles.hpp"

using namespace std;

#define TOM_TESTING
//#define TOM_DOUBLE

#ifdef TOM_DOUBLE
	typedef double tfloat;
	typedef cufftDoubleComplex tcomplex;
	#define IS_TFLOAT_DOUBLE true
	#define cmul cuCmul
	#define cconj cuConj
#else
	typedef float tfloat;
	typedef cufftComplex tcomplex;
	#define IS_TFLOAT_DOUBLE false
	#define cmul cuCmulf
	#define cconj cuConjf
#endif

struct tfloat2
{	
	tfloat x;
	tfloat y;

	__host__ __device__ tfloat2(tfloat x, tfloat y) : x(x), y(y) {}
};

struct tfloat3
{
	tfloat x;
	tfloat y;
	tfloat z;

	tfloat3(tfloat x, tfloat y, tfloat z) : x(x), y(y), z(z) {}
	tfloat3(int x, int y, int z) : x((tfloat)x), y((tfloat)y), z((tfloat)z) {}
	tfloat3(tfloat val) : x(val), y(val), z(val) {}
};

struct tfloat4
{
	tfloat x;
	tfloat y;
	tfloat z;
	tfloat w;

	tfloat4(tfloat x, tfloat y, tfloat z, tfloat w) : x(x), y(y), z(z), w(w) {}
};

struct tfloat5
{
	tfloat x;
	tfloat y;
	tfloat z;
	tfloat w;
	tfloat v;

	tfloat5(tfloat x, tfloat y, tfloat z, tfloat w, tfloat v) : x(x), y(y), z(z), w(w), v(v) {}
};

inline int2 toInt2(int x, int y)
{
	int2 value = {x, y};
	return value;
}

inline int3 toInt3(int x, int y, int z)
{
	int3 value = {x, y, z};
	return value;
}

inline uint3 toUint3(uint x, uint y, uint z)
{
	uint3 value = {x, y, z};
	return value;
}

inline uint3 toUint3(int x, int y, int z)
{
	uint3 value = {(uint)x, (uint)y, (uint)z};
	return value;
}

inline ushort3 toShort3(int x, int y, int z)
{
	ushort3 value = {(ushort)x, (ushort)y, (ushort)z};
}

inline int3 toInt3(int2 val)
{
	int3 value = {val.x, val.y, 1};
	return value;
}

struct imgstats5
{
	tfloat mean;
	tfloat min;
	tfloat max;
	tfloat stddev;
	tfloat var;

	imgstats5(tfloat mean, tfloat min, tfloat max, tfloat stddev, tfloat var) : mean(mean), min(min), max(max), stddev(stddev), var(var) {}
	imgstats5() : mean(0), min(0), max(0), stddev(0), var(0) {}
};

template <class T1, class T2> struct tuple2
{
	T1 t1;
	T2 t2;

	__host__ __device__ tuple2(T1 t1, T2 t2) : t1(t1), t2(t2) {}
	__host__ __device__ tuple2() {}
};

typedef unsigned int uint;

#define PI 3.1415926535897932384626433832795
#define PI2 6.283185307179586476925286766559

#define getOffset(x, y, stride) ((y) * (stride) + (x))
#define getZigzag(x, stride) abs((((x) - (stride)) % ((stride) * 2)) - (stride))
#define DimensionCount(dims) (3 - max(2 - max((dims).z, 1), 0) - max(2 - max((dims).y, 1), 0) - max(2 - max((dims).x, 1), 0))
#define NextMultipleOf(value, base) (((value) + (base) - 1) / (base) * (base))
#define Elements(dims) ((dims).x * (dims).y * (dims).z)
#define ElementsFFT(dims) (((dims).x / 2 + 1) * (dims).y * (dims).z)

#define min(x, y) ((x) > (y) ? (y) : (x))
#define max(x, y) ((x) < (y) ? (y) : (x))

enum T_INTERP_MODE 
{ 
	T_INTERP_LINEAR = 1,
	T_INTERP_CUBIC = 2,
	T_INTERP_FOURIER = 3
};

enum EM_DATATYPE
{
	EM_BYTE = 1,
	EM_SHORT = 2,
	EM_SHORTCOMPLEX = 3,
	EM_LONG = 4,
	EM_SINGLE = 5,
	EM_SINGLECOMPLEX = 8,
	EM_DOUBLE = 9,
	EM_DOUBLECOMPLEX = 10
};


// Define this to turn on error checking
#define CUDA_ERROR_CHECK

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
	if ( cudaSuccess != err )
		printf(cudaGetErrorString( err ));
#endif
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
	cudaError err = cudaGetLastError();
	if ( cudaSuccess != err )
		printf(cudaGetErrorString( err ));
#endif
}

/**
 * \brief Executes a call and prints the time needed for execution.
 * \param[in] call	The call to be executed
 */
#ifdef TOM_TESTING
	#define CUDA_MEASURE_TIME(call) \
			{ \
				float time = 0.0f; \
				cudaEvent_t start, stop; \
				cudaEventCreate(&start); \
				cudaEventCreate(&stop); \
				cudaEventRecord(start); \
				call; \
				cudaDeviceSynchronize(); \
				cudaEventRecord(stop); \
				cudaEventSynchronize(stop); \
				cudaEventElapsedTime(&time, start, stop); \
				printf("Kernel in %s executed in %f ms.\n", __FILE__, time); \
			}
#else
	#define CUDA_MEASURE_TIME(call) call
#endif

#endif