#ifndef DEVICE_FUNCTIONS
#define DEVICE_FUNCTIONS

inline __device__ float sinc(float x)
{
	if (x == 0.0f)
		return 1.0f;
	else
		return sin(x * PI) / (x * PI);
}

inline __device__ double sinc(double x)
{
	if (x == 0.0)
		return 1.0;
	else
		return sin(x * 3.1415926535897932384626433832795) / (x * 3.1415926535897932384626433832795);
}

#endif