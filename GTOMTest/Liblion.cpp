#include "Prerequisites.h"
#include "liblion.h"

TEST(Relion, Liblion)
{
	cudaDeviceReset();

	//Case 1:
	/*{
		relion::HealpixSampling sampling;
		sampling.setTranslations(1, 0);
		sampling.setOrientations(3);
		sampling.psi_step = -1;
		sampling.limit_tilt = -91;
		sampling.initialise(NOPRIOR, 3);
	}*/

	//Case 2:
	{
#pragma omp parallel for
		for (int i = 0; i < 10; i++)
		{
			relion::BackProjector backprojector(160, 3, "C1");
			backprojector.initZeros(160);

			relion::FourierTransformer transformer;
			relion::Matrix2D<float> A3D;

			relion::MultidimArray<float> I2D, vol;
			relion::MultidimArray<relion::Complex > F2D, F2Dp;
			relion::MultidimArray<float> Fweight;

			I2D.resize(160, 160);
			I2D.initConstant(0);
			I2D.data[80 * 160 + 80] = 1.0;

			relion::CenterFFT(I2D, true);
			transformer.FourierTransform(I2D, F2Dp);
			relion::windowFourierTransform(F2Dp, F2D, 160);

			Fweight.resize(F2D);
			Fweight.initConstant(1);

			Euler_angles2matrix(0, 0, 0, A3D);

			backprojector.set2DFourierTransform(F2D, A3D, IS_NOT_INV, &Fweight);


			relion::MultidimArray<float> dummy;
			backprojector.reconstruct(vol, 1, false, 1., dummy, dummy, dummy, dummy, 1, false, false, 1);

			/*float* h_vol = (float*)malloc(160 * 160 * 160 * sizeof(float));
			for (int i = 0; i < 160 * 160 * 160; i++)
			h_vol[i] = (float)vol.data[i];

			WriteMRC(h_vol, toInt3(160, 160, 160), "h_relion_rec.mrc");*/
		}
	}

	cudaDeviceReset();
}
