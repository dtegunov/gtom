#include "Prerequisites.h"

TEST(Helpers, Euler)
{
	cudaDeviceReset();

	//Case 1:
	{
		tfloat3 angeuler = tfloat3(ToRad(0.0f), ToRad(50.0f), ToRad(20.0f));
		tfloat3 angpolar = tfloat3(0.0f, ToRad(20.0f), ToRad(30.0f));

		glm::mat3 mateuler = Matrix3Euler(angeuler);
		glm::mat3 matpolar = Matrix3PolarViewVector(angpolar, ToRad(0.0f));
		
		cout << mateuler[0][0];
	}

	cudaDeviceReset();
}