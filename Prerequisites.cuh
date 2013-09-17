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

#define PI 3.14159265358979f

#define getOffset(x, y, stride) ((y) * (stride) + (x))
#define getZigzag(x, stride) abs((((x) - (stride)) % ((stride) * 2)) - (stride))

#define min(x, y) x > y ? y : x
#define max(x, y) x < y ? y : x

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