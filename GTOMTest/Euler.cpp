#include "Prerequisites.h"

TEST(Helpers, Euler)
{
	cudaDeviceReset();

	//Case 1:
	{
		tfloat3 anglesori = tfloat3(0.1f, 0.5f, 2.0f);
		glm::mat4 mori = Matrix4Euler(anglesori);
		tfloat3 anglesrec = EulerFromMatrix(mori);
		glm::mat4 mrec = Matrix4Euler(anglesrec);
		tfloat3 anglesrev = EulerFromMatrix(glm::inverse(mori));

		glm::vec4 transori = mori * glm::vec4(1, 2, 3, 1);
		glm::vec4 transrec = mrec * glm::vec4(1, 2, 3, 1);

		tfloat3 angles = tfloat3(0.0f, 0.1f, PIHALF);
		glm::mat4 m = Matrix4Euler(angles);
		glm::vec4 v = m * glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);

		cout << anglesori.x;
	}

	cudaDeviceReset();
}