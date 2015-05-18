#include "Prerequisites.cuh"
#include "CTF.cuh"


namespace gtom
{
	//Estimates the highest frequency that can be discretized on the give grid without aliasing
	//Adapted from "CTER—Rapid estimation of CTF parameters with error assessment", Penczek et al. 2014

	uint CTFGetAliasingCutoff(CTFParams params, uint sidelength)
	{
		double aliasinglevel = 1.0 / (params.pixelsize * 1e10 * (double)(sidelength / 2 + 1));
		double lambda = CTFParamsLean(params).lambda * 1e10;

		for (uint x = 1; x < sidelength / 2; x++)
		{
			double s = (double)x / ((double)(sidelength / 2 * 2) * params.pixelsize * 1e10);

			double a = 0.5 * params.defocus * 1e10 * lambda;
			double b = 0.25 * params.Cs * 1e10 * lambda * lambda * lambda;

			double factors[5];
			factors[4] = b;
			factors[3] = 4.0 * b * s;
			factors[2] = 6.0 * b * s * s - a;
			factors[1] = 4.0 * b * s * s * s - 2.0 * a * s;
			factors[0] = -1.0;

			int numroots = 0;
			Polynomial poly;
			poly.SetCoefficients(factors, 4);
			double rootsRe[10], rootsIm[10];
			poly.FindRoots(rootsRe, rootsIm, &numroots);

			double smallest = 1e30;
			for (uint i = 0; i < numroots; i++)
				smallest = min(smallest, sqrt(rootsRe[i] * rootsRe[i] + rootsIm[i] * rootsIm[i]));

			if (smallest < aliasinglevel)
				return x - 1;
		}

		return sidelength / 2;
	}
}