#ifndef DEVICE_FUNCTIONS_CUH
#define DEVICE_FUNCTIONS_CUH

#define GLM_FORCE_RADIANS
#define GLM_FORCE_INLINE
#define GLM_FORCE_CUDA
#include "glm/glm.hpp"

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

	inline __device__ glm::mat3 d_Matrix3Euler(float3 angles)
	{
		float alpha = angles.x;
		float beta = angles.y;
		float gamma = angles.z;

		float ca, sa, cb, sb, cg, sg;
		float cc, cs, sc, ss;

		ca = cos(alpha);
		cb = cos(beta);
		cg = cos(gamma);
		sa = sin(alpha);
		sb = sin(beta);
		sg = sin(gamma);
		cc = cb * ca;
		cs = cb * sa;
		sc = sb * ca;
		ss = sb * sa;

		return glm::mat3(cg * cc - sg * sa, -sg * cc - cg * sa, sc,
						 cg * cs + sg * ca, -sg * cs + cg * ca, ss,
								  -cg * sb,			   sg * sb, cb);
	}

	inline __device__ glm::mat3 d_Matrix3RotationX(float angle)
	{
		float c = cos(angle);
		float s = sin(angle);

		return glm::mat3(1, 0, 0, 0, c, s, 0, -s, c);
	}

	inline __device__ glm::mat3 d_Matrix3RotationY(float angle)
	{
		float c = cos(angle);
		float s = sin(angle);

		return glm::mat3(c, 0, -s, 0, 1, 0, s, 0, c);
	}

	inline __device__ glm::mat3 d_Matrix3RotationZ(float angle)
	{
		float c = cos(angle);
		float s = sin(angle);

		return glm::mat3(c, s, 0, -s, c, 0, 0, 0, 1);
	}
}
#endif