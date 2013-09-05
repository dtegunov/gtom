#pragma once

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cufft.h>
#include <cublas_v2.h>

#define GLM_FORCE_RADIANS
#include "glm\glm.hpp"
#include "glm\gtc\matrix_transform.hpp"
#include "glm\gtx\quaternion.hpp"
#include "glm\gtx\euler_angles.hpp"

using namespace std;

#define PI 3.14159265358979f


#define getOffset(x, y, stride) ((y) * (stride) + (x))
#define getZigzag(x, stride) abs((((x) - (stride)) % ((stride) * 2)) - (stride))

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