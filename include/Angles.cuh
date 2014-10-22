#include "Prerequisites.cuh"

#define GLM_FORCE_RADIANS
#define GLM_FORCE_INLINE
#define GLM_FORCE_CUDA
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtx/quaternion.hpp"
#include "glm/gtx/euler_angles.hpp"
#include "glm/gtc/type_ptr.hpp"

glm::mat4 GetEulerRotation(tfloat3 angles);
glm::mat4 GetEulerRotation(tfloat2 angles);
glm::mat3 GetEulerRotation3(tfloat3 angles);
glm::mat2 Get2DRotation(tfloat angle);

tfloat3* GetEqualAngularSpacing(tfloat2 phirange, tfloat2 thetarange, tfloat2 psirange, tfloat increment, int &numangles);