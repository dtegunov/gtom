#include "Prerequisites.h"

TEST(CTF, Simulation)
{
	cudaDeviceReset();

	//Case 1:
	{
		int2 dimspolar = toInt2(1024, 1);

		float2* h_ps2dcoords = (float2*)malloc(Elements2(dimspolar) * sizeof(float2));
		float invhalfsize = 2.0f / (dimspolar.x * 2);
		float anglestep = PI / (float)dimspolar.y;
		for (int a = 0; a < dimspolar.y; a++)
		{
			float angle = (float)a * anglestep + PIHALF;
			for (int r = 0; r < dimspolar.x; r++)
				h_ps2dcoords[a * dimspolar.x + r] = make_float2((float)r * invhalfsize, angle);
		}
		float2* d_ps2dcoords = (float2*)CudaMallocFromHostArray(h_ps2dcoords, Elements2(dimspolar) * sizeof(float2));

		CTFParams p;
		p.defocus = -3e-6f;
		//p.defocusdelta = 1e-6f;
		p.pixelsize = 2.96e-10f;
		
		tfloat* d_ps = CudaMallocValueFilled(Elements2(dimspolar), (tfloat)0);

		d_CTFSimulate(&p, d_ps2dcoords, d_ps, Elements2(dimspolar));

		d_WriteMRC(d_ps, toInt3(dimspolar), "d_ctf.mrc");
	}

	cudaDeviceReset();
}