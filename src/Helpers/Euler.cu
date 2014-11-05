#include "Prerequisites.cuh"
#include "Angles.cuh"

glm::mat4 Matrix4Euler(tfloat3 angles)
{
	float phi = angles.x;
	float theta = angles.y;
	float psi = angles.z;

	return Matrix4RotationZ(phi) * Matrix4RotationY(theta) * Matrix4RotationZ(psi);
}

glm::mat3 Matrix3Euler(tfloat3 angles)
{
	float phi = angles.x;
	float theta = angles.y;
	float psi = angles.z;

	return Matrix3RotationZ(phi) * Matrix3RotationY(theta) * Matrix3RotationZ(psi);
}

glm::mat4 Matrix4EulerLegacy(tfloat2 angles)
{
	float phi = PI / 2.0f - angles.x;
	float psi = angles.x - PI / 2.0f;
	float theta = angles.y;
	
	return glm::transpose(Matrix4Euler(tfloat3(phi, theta, psi)));
}

glm::mat4 Matrix4Translation(tfloat3 translation)
{
	return glm::mat4(1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0,  translation.x, translation.y, translation.z, 1);
}

glm::mat4 Matrix4Scale(tfloat3 scale)
{
	return glm::mat4(scale.x, 0, 0, 0,  0, scale.y, 0, 0,  0, 0, scale.z, 0,  0, 0, 0, 1);
}

glm::mat4 Matrix4RotationX(tfloat angle)
{
	double c = cos(angle);
	double s = sin(angle);

	return glm::mat4(1, 0, 0, 0,  0, c, s, 0,  0, -s, c, 0,  0, 0, 0, 1);
}

glm::mat4 Matrix4RotationY(tfloat angle)
{
	double c = cos(angle);
	double s = sin(angle);

	return glm::mat4(c, 0, -s, 0,  0, 1, 0, 0,  s, 0, c, 0,  0, 0, 0, 1);
}

glm::mat4 Matrix4RotationZ(tfloat angle)
{
	double c = cos(angle);
	double s = sin(angle);

	return glm::mat4(c, s, 0, 0,  -s, c, 0, 0,  0, 0, 1, 0,  0, 0, 0, 1);
}

glm::mat3 Matrix3Translation(tfloat2 translation)
{
	return glm::mat3(1, 0, 0,  0, 1, 0,  translation.x, translation.y, 1);
}

glm::mat3 Matrix3Scale(tfloat3 scale)
{
	return glm::mat3(scale.x, 0, 0,  0, scale.y, 0,  0, 0, scale.z);
}

glm::mat3 Matrix3RotationX(tfloat angle)
{
	double c = cos(angle);
	double s = sin(angle);

	return glm::mat3(1, 0, 0,  0, c, s,  0, -s, c);
}

glm::mat3 Matrix3RotationY(tfloat angle)
{
	double c = cos(angle);
	double s = sin(angle);

	return glm::mat3(c, 0, -s,  0, 1, 0,  s, 0, c);
}

glm::mat3 Matrix3RotationZ(tfloat angle)
{
	double c = cos(angle);
	double s = sin(angle);

	return glm::mat3(c, s, 0,  -s, c, 0,  0, 0, 1);
}

glm::mat2 Matrix2Scale(tfloat2 scale)
{
	return glm::mat2(scale.x, 0,  0, scale.y);
}

glm::mat2 Matrix2Rotation(tfloat angle)
{
	double c = cos(angle);
	double s = sin(angle);

	return glm::mat2(c, s, -s, c);
}