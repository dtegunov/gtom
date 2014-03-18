#include "..\GLMFunctions.cuh"

glm::mat4 GetEulerRotation(tfloat3 angles)
{
	float phi = angles.x;
	float psi = angles.y;
	float theta = angles.z;

	float cosphi = cos(phi), sinphi = sin(phi);
	float cospsi = cos(psi), sinpsi = sin(psi);
	float costheta = cos(theta), sintheta = sin(theta);

	glm::mat4 rotationMat;

	rotationMat[0][0] = cospsi * cosphi - costheta * sinpsi * sinphi;
	rotationMat[1][0] = sinpsi * cosphi + costheta * cospsi * sinphi;
	rotationMat[2][0] = sintheta * sinphi;
	rotationMat[3][0] = 0.0f;
	rotationMat[0][1] = -cospsi * sinphi - costheta * sinpsi * cosphi;
	rotationMat[1][1] = -sinpsi * sinphi + costheta * cospsi * cosphi;
	rotationMat[2][1] = sintheta * cosphi;
	rotationMat[3][1] = 0.0f;
	rotationMat[0][2] = sintheta * sinpsi;
	rotationMat[1][2] = -sintheta * cospsi;
	rotationMat[2][2] = costheta;
	rotationMat[3][2] = 0.0f;
	rotationMat[0][3] = 0.0f;
	rotationMat[1][3] = 0.0f;
	rotationMat[2][3] = 0.0f;
	rotationMat[3][3] = 1.0f;

	return rotationMat;
}

glm::mat4 GetEulerRotation(tfloat2 angles)
{
	float phi = PI / 2.0f - angles.x;
	float psi = angles.x - PI / 2.0f;
	float theta = angles.y;
	
	return glm::inverse(GetEulerRotation(tfloat3(phi, psi, theta)));
}