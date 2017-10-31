#include "..\Prerequisites.h"


void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
    char const * const errId = "GTOM:Resolution:AnisotropicFSCMap:InvalidInput";
    char const * const errMsg = "Invalid input to MEX file.";

    mxInitGPU();

	if (nrhs < 5)
        mexErrMsgIdAndTxt(errId, errMsg);

	mxArrayAdapter v1(prhs[0]);
	mxArrayAdapter v2(prhs[1]);
	int ndims = mxGetNumberOfDimensions(v1.underlyingarray);
	int3 dimsvolume = MWDimsToInt3(ndims, mxGetDimensions(v1.underlyingarray));
	tfloat anglestep = (tfloat)((double*)mxGetData(prhs[2]))[0];
	int minshell = (int)((double*)mxGetData(prhs[3]))[0];
	tfloat threshold = (tfloat)((double*)mxGetData(prhs[4]))[0];

	int nshells = dimsvolume.x / 2;
	int nphi = ceil(360.0f / (anglestep / 16));
	int ntheta = ceil(90.0f / (anglestep / 16));

	tfloat stepphi = 360.0f / (nphi - 1);
	tfloat steptheta = 90.0f / (ntheta - 1);

	tfloat conesigma = anglestep / 2 / 180 * PI;
	conesigma = 2.0f * conesigma * conesigma;

	tfloat dotcutoff = cos(anglestep * 1.5f / 180.0f * PI);

	int2 dimsresmap = toInt2(ntheta, nphi);
	int nangles = Elements2(dimsresmap);
	tfloat* h_resmap = (tfloat*)malloc(nangles * sizeof(tfloat));

	tfloat* h_filter = MallocValueFilled(Elements(dimsvolume), (tfloat)0);

	float3* h_angledirs = (float3*)malloc(nangles * sizeof(float3));
	for (int p = 0, a = 0; p < nphi; p++)
	{
		float phi = p * stepphi / 180.0f * PI;
		for (int t = 0; t < ntheta; t++, a++)
		{
			float theta = t * steptheta / 180.0f * PI;

			h_angledirs[a] = make_float3(cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta));
		}
	}

	tfloat* h_nums = MallocValueFilled(nangles * nshells, (tfloat)0);
	tfloat* h_denoms1 = MallocValueFilled(nangles * nshells, (tfloat)0);
	tfloat* h_denoms2 = MallocValueFilled(nangles * nshells, (tfloat)0);
	tfloat* h_weightsums = MallocValueFilled(nangles * nshells, (tfloat)0);

	tcomplex* h_volumeft1 = v1.GetAsManagedTComplex();
	tcomplex* h_volumeft2 = v2.GetAsManagedTComplex();
	
	for (int z = 0; z < dimsvolume.z; z++)
	{
		int zz = z - dimsvolume.x / 2;
		float zz2 = zz * zz;
		for (int y = 0; y < dimsvolume.y; y++)
		{
			int yy = y - dimsvolume.y / 2;
			float yy2 = yy * yy;
			for (int x = 0; x < dimsvolume.x / 2 + 1; x++)
			{
				int xx = x - dimsvolume.x / 2;
				float xx2 = xx * xx;
				float r = sqrt(zz2 + yy2 + xx2);
				int ri = (int)(r + 0.5f);
				if (ri >= nshells)
					continue;

				float3 dir = make_float3(xx / r, yy / r, zz / r);

				tcomplex val1 = h_volumeft1[(z * dimsvolume.y + y) * dimsvolume.x + x];
				tcomplex val2 = h_volumeft2[(z * dimsvolume.y + y) * dimsvolume.x + x];

				tfloat num = dotp2(val1, val2);
				tfloat denom1 = dotp2(val1, val1);
				tfloat denom2 = dotp2(val2, val2);

				#pragma omp parallel for
				for (int a = 0; a < nangles; a++)
				{
					float dot = abs(dotp(dir, h_angledirs[a]));
					if (dot < dotcutoff)
						continue;

					float anglediff = acos(dot);

					float weight = exp(-(anglediff * anglediff) / conesigma);

					h_nums[a * nshells + ri] += num * weight;
					h_denoms1[a * nshells + ri] += denom1 * weight;
					h_denoms2[a * nshells + ri] += denom2 * weight;
					h_weightsums[a * nshells + ri] += weight;
				}
			}
		}

		if ((z + 1) % 10 == 0)
		{
			mexPrintf("%f\n", (float)(z + 1) / dimsvolume.x * 100.0f);
			//mexEvalString("drawnow;");
			//mexEvalString("disp('bla')");
			mexEvalString("pause(.001);");
		}
	}

	for (int a = 0, i = 0; a < nangles; a++)
	{
		bool foundlimit = false;
		for (int s = 0; s < nshells; s++, i++)
		{
			if (s < minshell || h_weightsums[i] < 6.0f)
				h_nums[i] = 1;
			else
			{
				tfloat weightsum = h_weightsums[i];
				h_nums[i] = h_nums[i] / weightsum / sqrt(h_denoms1[i] / weightsum * (h_denoms2[i] / weightsum));
			}

			if (!foundlimit && h_nums[i] < threshold)
			{
				foundlimit = true;
				float current = h_nums[i - 1];
				float next = h_nums[i];
				
				h_resmap[a] = tmax(1, (tfloat)(s - 1) + tmax(tmin((threshold - current) / (next - current + (tfloat)0.00001), 1.0f), 0.0f));
			}
		}

		if (!foundlimit)
			h_resmap[a] = nshells - 1;
	}

	for (int z = 0; z < dimsvolume.z; z++)
	{
		for (int y = 0; y < dimsvolume.y; y++)
		{
			for (int x = 0; x < dimsvolume.x; x++)
			{
				int zz = z - dimsvolume.x / 2;
				float zz2 = zz * zz;
				int yy = y - dimsvolume.y / 2;
				float yy2 = yy * yy;
				int xx = x - dimsvolume.x / 2;
				float xx2 = xx * xx;

				float r = sqrt(zz2 + yy2 + xx2);

				int ri = (int)(r + 0.5f);
				if (ri >= nshells)
					continue;

				if (zz < 0)
				{
					zz = -zz;
					yy = -yy;
					xx = -xx;
				}

				float3 dir = make_float3(xx / r, yy / r, zz / r);
				float theta = acos(dir.z) / PI * 180.0f;
				float phi = atan2(dir.y, dir.x) / PI * 180.0f;

				if (phi < 0)
					phi += 360;

				float p = tmin(phi / stepphi, nphi - 1);
				float t = tmin(theta / steptheta, ntheta - 1);

				int p0 = floor(p);
				int p1 = tmin(nphi - 1, ceil(p));
				int t0 = floor(t);
				int t1 = tmin(ntheta - 1, ceil(t));

				float f00 = h_resmap[p0 * ntheta + t0];
				float f01 = h_resmap[p0 * ntheta + t1];
				float f10 = h_resmap[p1 * ntheta + t0];
				float f11 = h_resmap[p1 * ntheta + t1];

				float f0 = lerp(f00, f01, t - t0);
				float f1 = lerp(f10, f11, t - t0);
				float f = lerp(f0, f1, p - p0);

				float diff = r - f;
				h_filter[(z * dimsvolume.y + y) * dimsvolume.x + x] = 1 - tmax(0, tmin(1, diff));
			}
		}
	}

	mwSize mapsize[3];
	mapsize[0] = (mwSize)dimsresmap.x;
	mapsize[1] = (mwSize)dimsresmap.y;
	mapsize[2] = (mwSize)1;
	mxArrayAdapter B(mxCreateNumericArray(2,
					 mapsize,
					 mxGetClassID(v1.underlyingarray),
					 mxREAL));
	B.SetFromTFloat(h_resmap);
	plhs[0] = B.underlyingarray;

	mwSize mapsize2[3];
	mapsize2[0] = (mwSize)dimsvolume.x;
	mapsize2[1] = (mwSize)dimsvolume.y;
	mapsize2[2] = (mwSize)dimsvolume.z;
	mxArrayAdapter C(mxCreateNumericArray(3,
					 mapsize2,
					 mxGetClassID(v1.underlyingarray),
					 mxREAL));
	C.SetFromTFloat(h_filter);
	plhs[1] = C.underlyingarray;

	free(h_filter);
	free(h_resmap);
	free(h_nums);
	free(h_denoms1);
	free(h_denoms2);
	free(h_weightsums);
	free(h_angledirs);
}