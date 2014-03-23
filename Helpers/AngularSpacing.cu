#include "../GLMFunctions.cuh"

tfloat3* GetEqualAngularSpacing(tfloat2 phirange, tfloat2 thetarange, tfloat2 psirange, tfloat increment, int &numangles)
{
	int count = 0;
	int rot_nstep = 0;

	for(float tilt = thetarange.x; tilt <= thetarange.y - increment; tilt += increment)
	{
		if(tilt > 0)
			rot_nstep = (int)ceil(PI2 * sin(tilt) / increment);
		else 
			rot_nstep = 1;

		float rot_sam = PI2 / (float)rot_nstep;

		for(float rot = phirange.x; rot <= phirange.y - rot_sam; rot += rot_sam)
			for(float psi = psirange.x; psi <= psirange.y - increment; psi += increment)
				count++;
	}

	tfloat3* h_angles = (tfloat3*)malloc(count * sizeof(tfloat3));
	numangles = count;
	count = 0;

	for(float tilt = thetarange.x; tilt <= thetarange.y - increment; tilt += increment)
	{
		if(tilt > 0)
			rot_nstep = (int)ceil(PI2 * sin(tilt) / increment);
		else 
			rot_nstep = 1;

		float rot_sam = PI2 / (float)rot_nstep;

		for(float rot = phirange.x; rot <= phirange.y - rot_sam; rot += rot_sam)
			for(float psi = psirange.x; psi <= psirange.y - increment; psi += increment)
				h_angles[count++] = tfloat3(rot, tilt, psi);
	}

	return h_angles;
}