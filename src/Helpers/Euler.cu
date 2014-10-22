#include "Prerequisites.cuh"
#include "Angles.cuh"

glm::mat4 GetEulerRotation(tfloat3 angles)
{
	float phi = angles.x;
	float theta = angles.y;
	float psi = angles.z;

	float cosphi = cos(phi), sinphi = sin(phi);
	float cospsi = cos(psi), sinpsi = sin(psi);
	float costheta = cos(theta), sintheta = sin(theta);

	glm::mat4 rotationMat;

	//IMPORTANT: mat[x][y] points to y'th element of the x'th column according to GLSL specification

	rotationMat[0][0] = cospsi * cosphi - costheta * sinpsi * sinphi;
	rotationMat[0][1] = sinpsi * cosphi + costheta * cospsi * sinphi;
	rotationMat[0][2] = sintheta * sinphi;
	rotationMat[0][3] = 0.0f;
	rotationMat[1][0] = -cospsi * sinphi - costheta * sinpsi * cosphi;
	rotationMat[1][1] = -sinpsi * sinphi + costheta * cospsi * cosphi;
	rotationMat[1][2] = sintheta * cosphi;
	rotationMat[1][3] = 0.0f;
	rotationMat[2][0] = sintheta * sinpsi;
	rotationMat[2][1] = -sintheta * cospsi;
	rotationMat[2][2] = costheta;
	rotationMat[2][3] = 0.0f;
	rotationMat[3][0] = 0.0f;
	rotationMat[3][1] = 0.0f;
	rotationMat[3][2] = 0.0f;
	rotationMat[3][3] = 1.0f;

	return rotationMat;
}

glm::mat3 GetEulerRotation3(tfloat3 angles)
{
	float phi = angles.x;
	float theta = angles.y;
	float psi = angles.z;

	float cosphi = cos(phi), sinphi = sin(phi);
	float cospsi = cos(psi), sinpsi = sin(psi);
	float costheta = cos(theta), sintheta = sin(theta);

	glm::mat3 rotationMat;

	//Result of Rz(psi) * Rx(theta) * Rz(phi)
	//IMPORTANT: mat[x][y] points to y'th element of the x'th column according to GLSL specification

	rotationMat[0][0] = cospsi * cosphi - costheta * sinpsi * sinphi;
	rotationMat[0][1] = sinpsi * cosphi + costheta * cospsi * sinphi;
	rotationMat[0][2] = sintheta * sinphi;
	rotationMat[1][0] = -cospsi * sinphi - costheta * sinpsi * cosphi;
	rotationMat[1][1] = -sinpsi * sinphi + costheta * cospsi * cosphi;
	rotationMat[1][2] = sintheta * cosphi;
	rotationMat[2][0] = sintheta * sinpsi;
	rotationMat[2][1] = -sintheta * cospsi;
	rotationMat[2][2] = costheta;

	return rotationMat;
}

glm::mat4 GetEulerRotation(tfloat2 angles)
{
	float phi = PI / 2.0f - angles.x;
	float psi = angles.x - PI / 2.0f;
	float theta = angles.y;
	
	return glm::inverse(GetEulerRotation(tfloat3(phi, theta, psi)));
}

glm::mat2 Get2DRotation(tfloat angle)
{
	float cosangle = cos(angle), sinangle = sin(angle);
	glm::mat2 rotationMat = glm::mat2(cosangle, sinangle, -sinangle, cosangle);

	return rotationMat;
}