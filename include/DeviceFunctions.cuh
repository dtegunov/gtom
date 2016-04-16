#ifndef DEVICE_FUNCTIONS_CUH
#define DEVICE_FUNCTIONS_CUH

namespace gtom
{
	inline __device__ float sinc(float x)
	{
		if (abs(x) <= 1e-4f)
			return 1.0f;
		else
			return sin(x * PI) / (x * PI);
	}

	inline __device__ double sinc(double x)
	{
		if (abs(x) <= 1e-8)
			return 1.0;
		else
			return sin(x * 3.1415926535897932384626433832795) / (x * 3.1415926535897932384626433832795);
	}
}
#endif