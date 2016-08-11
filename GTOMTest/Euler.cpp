#include "Prerequisites.h"

TEST(Helpers, Euler)
{
	cudaDeviceReset();

	//Case 1:
	{
		tfloat3 angeuler = tfloat3(ToRad(-13.0f), ToRad(-20.0f), ToRad(-161.0f));
		tfloat3 angpolar = tfloat3(0.0f, ToRad(20.0f), ToRad(30.0f));

		glm::mat3 mateuler = Matrix3Euler(angeuler);
		glm::mat3 matpolar = Matrix3PolarViewVector(angpolar, ToRad(0.0f));
		
		cout << mateuler[0][0];

		glm::mat3 mataroundy = Matrix3Euler(tfloat3(0.0f, ToRad(20.0f), 0.0f));
		glm::vec3 towardsx = glm::vec3(1, 0, 0);
		glm::vec3 transvec = mataroundy * towardsx;

		cout << transvec.x;
	}

	cudaDeviceReset();
}