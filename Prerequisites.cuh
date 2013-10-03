#pragma once

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
#else
	typedef float tfloat;
	typedef cufftComplex tcomplex;
	#define IS_TFLOAT_DOUBLE false
#endif

struct tfloat2
{
	tfloat x;
	tfloat y;

	tfloat2(tfloat x, tfloat y) : x(x), y(y) {}
};

struct tfloat3
{
	tfloat x;
	tfloat y;
	tfloat z;

	tfloat3(tfloat x, tfloat y, tfloat z) : x(x), y(y), z(z) {}
	tfloat3(int x, int y, int z) : x((tfloat)x), y((tfloat)y), z((tfloat)z) {}
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

#define PI 3.14159265358979
#define PI2 6.28318530717959

#define getOffset(x, y, stride) ((y) * (stride) + (x))
#define getZigzag(x, stride) abs((((x) - (stride)) % ((stride) * 2)) - (stride))

#define min(x, y) ((x) > (y) ? (y) : (x))
#define max(x, y) ((x) < (y) ? (y) : (x))

//Utility class used to avoid linker errors with extern
//unsized shared memory arrays with templated type
template<class T> struct SharedMemory
{
    __device__ inline operator T*()
    {
        extern __shared__ int __smem[];
        return (T*)__smem;
    }

    __device__ inline operator const T*() const
    {
        extern __shared__ int __smem[];
        return (T*)__smem;
    }
};

//specialize for tfloat to avoid unaligned memory
//access compile errors
template<> struct SharedMemory<tfloat>
{
    __device__ inline operator tfloat*()
    {
        extern __shared__ tfloat __smem_d[];
        return (tfloat*)__smem_d;
    }

    __device__ inline operator const tfloat*() const
    {
        extern __shared__ tfloat __smem_d[];
        return (tfloat*)__smem_d;
    }
};

// Define this to turn on error checking
#define CUDA_ERROR_CHECK

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
	if ( cudaSuccess != err )
	{
		printf(cudaGetErrorString( err ));
		//exit( -1 );
	}
#endif

	return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
	cudaError err = cudaGetLastError();
	if ( cudaSuccess != err )
	{
		//printf( "cudaCheckError() failed at %s:%i : %s\n",
				 //file, line, cudaGetErrorString( err ) );
		printf(cudaGetErrorString( err ));
		//exit( -1 );
	}

	// More careful checking. However, this will affect performance.
	// Comment away if needed.
	/*err = cudaDeviceSynchronize();
	if( cudaSuccess != err )
	{
		printf( "cudaCheckError() with sync failed at %s:%i : %s\n",
				 file, line, cudaGetErrorString( err ) );
		//exit( -1 );
	}*/
#endif

	return;
}

/**
 * \brief Executes the call, synchronizes the device and puts the ellapsed time into 'time'.
 * \param[in] call	The call to be executed
 * \param[in] time	Measured time will be written here
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