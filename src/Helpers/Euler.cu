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

tfloat3 EulerFromMatrix(glm::mat4 m)
{
	// In glm, m[x][y] is element at column x, row y

	float phi = 0.0f, theta = 0.0f, psi = 0.0f;
	float abssintheta = sqrt(m[2][0] * m[2][0] + m[2][1] * m[2][1]);
	if (abssintheta > 0.00001f)
	{
		psi = PI - atan2(m[1][2], m[0][2]);
		phi = PI - atan2(m[2][1], -m[2][0]);
		float s;
		if (sin(phi) == 0.0f)
			s = -m[2][0] / cos(phi) >= 0.0f ? 1.0f : -1.0f;
		else
			s = m[2][1] / sin(phi) >= 0.0f ? 1.0f : -1.0f;
		theta = atan2(s * abssintheta, m[2][2]);
	}
	else
	{
		psi = 0.0f;
		if (m[2][2] > 0.0f)
		{
			theta = 0.0f;
			phi = atan2(m[0][1], m[0][0]);
		}
		else
		{
			theta = PI;
			phi = atan2(m[0][1], m[0][0]);
		}
	}

	return tfloat3(phi, theta, psi);
}

tfloat3 EulerFromMatrix(glm::mat3 m)
{
	// In glm, m[x][y] is element at column x, row y

	float phi = 0.0f, theta = 0.0f, psi = 0.0f;
	float abssintheta = sqrt(m[2][0] * m[2][0] + m[2][1] * m[2][1]);
	if (abssintheta > 0.00001f)
	{
		psi = PI - atan2(m[1][2], m[0][2]);
		phi = PI - atan2(m[2][1], -m[2][0]);
		float s;
		if (sin(phi) == 0.0f)
			s = -m[2][0] / cos(phi) >= 0.0f ? 1.0f : -1.0f;
		else
			s = m[2][1] / sin(phi) >= 0.0f ? 1.0f : -1.0f;
		theta = atan2(s * abssintheta, m[2][2]);
	}
	else
	{
		psi = 0.0f;
		if (m[2][2] > 0.0f)
		{
			theta = 0.0f;
			phi = atan2(m[0][1], m[0][0]);
		}
		else
		{
			theta = PI;
			phi = atan2(m[0][1], m[0][0]);
		}
	}

	return tfloat3(phi, theta, psi);
}

tfloat3 EulerInverse(tfloat3 angles)
{
	return tfloat3(-angles.z, -angles.y, -angles.x);
}

float EulerCompare(tfloat3 angles1, tfloat3 angles2)
{
	glm::mat3 m1 = Matrix3Euler(angles1);
	glm::mat3 m2 = Matrix3Euler(angles2);

	glm::vec3 v1 = glm::normalize(m1 * glm::vec3(1, 1, 1));
	glm::vec3 v2 = glm::normalize(m2 * glm::vec3(1, 1, 1));

	return glm::dot(v1, v2);
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