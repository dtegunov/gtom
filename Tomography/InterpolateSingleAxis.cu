#include "../Prerequisites.cuh"
#include "../Functions.cuh"


////////////////////////////
//CUDA kernel declarations//
////////////////////////////

//__global__ void InterpolateSingleAxisTiltKernel(tcomplex* d_projft, size_t elementsproj, int3 dims, tcomplex* d_interpolated, tfloat* d_factors, short* d_indices, short npoints);
__global__ void InterpolateSingleAxisTiltKernel(tcomplex* d_projft, size_t elementsproj, int3 dims, tcomplex* d_interpolated, tfloat* d_angles, tfloat interpangle, short* d_indices, short npoints, tfloat interpradius);


////////////////////////////////////////////////////////////////////
//Interpolates a certain image in a tilt series from its neighbors//
////////////////////////////////////////////////////////////////////

void d_InterpolateSingleAxisTilt(tcomplex* d_projft, int3 dimsproj, tcomplex* d_interpolated, tfloat* h_angles, int interpindex, int maxpoints, tfloat smoothsigma)
{
	tfloat interpangle = h_angles[interpindex];
	int npoints = dimsproj.z - 1;//maxpoints;
	tfloat* h_factors = (tfloat*)malloc(npoints * sizeof(tfloat));
	short* h_indices = (short*)malloc(npoints * sizeof(short));

	int n = 0;
	//for (int i = max(0, interpindex - maxpoints / 2); i <= min(dimsproj.z - 1, interpindex + maxpoints / 2); i++)
	for (int i = 0; i < dimsproj.z; i++)
	{
		bool isconj = false;
		int ii = i;
		if(i < 0 || i >= dimsproj.z)
		{
			isconj = true;
			ii = (i + dimsproj.z * maxpoints) % dimsproj.z;
		}
		double anglei = isconj ? h_angles[ii] - (i < 0 ? ToRad(180.0f) : ToRad(-180.0f)) : h_angles[ii];

		if(i == interpindex)
			continue;

		/*double factor = 1.0;
		for (int j = interpindex - maxpoints / 2; j <= interpindex + maxpoints / 2; j++)
		{
			bool isconj = false;
			int jj = j;
			if(j < 0 || j >= dimsproj.z)
			{
				isconj = true;
				jj = (j + dimsproj.z * maxpoints) % dimsproj.z;
			}
			double anglej = isconj ? h_angles[jj] - (j < 0 ? ToRad(180.0f) : ToRad(-180.0f)) : h_angles[jj];

			if(j == interpindex || j == i)
				continue;
			factor *= ((double)interpangle - anglej) / (anglei - anglej);
		}
		h_factors[n] = (tfloat)factor;*/
		h_factors[n] = anglei;

		h_indices[n] = (short)i;
		n++;
	}

	tfloat* d_factors = (tfloat*)CudaMallocFromHostArray(h_factors, n * sizeof(tfloat));
	short* d_indices = (short*)CudaMallocFromHostArray(h_indices, n * sizeof(short));
	free(h_factors);
	free(h_indices);

	int TpB = min(NextMultipleOf((dimsproj.x / 2 + 1) * dimsproj.y, 32), 128);
	dim3 grid = dim3(min(((dimsproj.x / 2 + 1) * dimsproj.y + TpB - 1) / TpB, 8192));
	//InterpolateSingleAxisTiltKernel <<<grid, TpB>>> (d_projft, (dimsproj.x / 2 + 1) * dimsproj.y, dimsproj, d_interpolated, d_factors, d_indices, n);
	InterpolateSingleAxisTiltKernel <<<grid, TpB>>> (d_projft, (dimsproj.x / 2 + 1) * dimsproj.y, dimsproj, d_interpolated, d_factors, interpangle, d_indices, n, smoothsigma);

	cudaFree(d_factors);
	cudaFree(d_indices);
}


////////////////
//CUDA kernels//
////////////////

//__global__ void InterpolateSingleAxisTiltKernel(tcomplex* d_projft, size_t elementsproj, int3 dims, tcomplex* d_interpolated, tfloat* d_factors, short* d_indices, short npoints)
//{
//	for(size_t id = blockIdx.x * blockDim.x + threadIdx.x; 
//		id < elementsproj; 
//		id += blockDim.x * gridDim.x)
//	{
//		int x = id % (dims.x / 2 + 1);
//		int y = id / (dims.x / 2 + 1);
//		int my = y;
//
//		double sumre = 0.0;
//		double sumim = 0.0;
//		for (int n = 0; n < npoints; n++)
//		{
//			int index = d_indices[n];
//			bool isconj = false;
//			my = y;
//			if(index < 0 || index >= dims.z)
//			{
//				isconj = true;
//				index = (index + npoints * dims.z) % dims.z;
//				if(x > 0 && y > 0)
//					my = dims.y - y;
//			}
//			double factor = (double)d_factors[n];
//			sumre += (double)d_projft[index * elementsproj + getOffset(x, my, dims.x / 2 + 1)].x * factor;
//			sumim += (double)d_projft[index * elementsproj + getOffset(x, my, dims.x / 2 + 1)].y * factor * ((isconj && !(x == 0 && y == 0)) ? -1.0f : 1.0f);
//		}
//		d_interpolated[id].x = (tfloat)sumre;
//		d_interpolated[id].y = (tfloat)sumim;
//	}
//}

__global__ void InterpolateSingleAxisTiltKernel(tcomplex* d_projft, size_t elementsproj, int3 dims, tcomplex* d_interpolated, tfloat* d_angles, tfloat interpangle, short* d_indices, short npoints, tfloat interpradius)
{
	for(size_t id = blockIdx.x * blockDim.x + threadIdx.x; 
		id < elementsproj; 
		id += blockDim.x * gridDim.x)
	{
		float x = (float)(id % (dims.x / 2 + 1));
		float y = (float)(id / (dims.x / 2 + 1));

		float sumre = 0.0f;
		float sumim = 0.0f;
		float samples = 0.0f;

		float coscenter = abs(cos(interpangle));
		float centerx = coscenter * x;
		float centerz = sin(interpangle) * x;

		for (int n = npoints - 1; n < npoints; n++)
		{
			int index = d_indices[n];
			float angle = d_angles[n];
			bool isconj = false;
			float anglediff = abs(interpangle - angle);
			if(abs(interpangle - (angle - PI)) < anglediff)
			{
				angle = angle - PI;
				isconj = true;
			}
			else if(abs(interpangle - (angle + PI)) < anglediff)
			{
				angle = angle + PI;
				isconj = true;
			}
			samples = angle;
			if(!(x > 0 && y > 0))
				isconj = false;
			int ylocal = isconj ? dims.y - (int)y : (int)y;

			float cosangle = abs(cos(angle));
			float scalefac = coscenter / cosangle;
			scalefac = 1.0f + (scalefac - 1.0f) * (1.0f - 32.0f / 128.0f);
			float dx = x * scalefac;
			float anglex = dx * cosangle - centerx;
			float anglez = dx * sin(angle) - centerz;

			float d = interpradius - sqrt(anglex * anglex + anglez * anglez);
			if(d <= 0.0f)
				continue;
			d *= max(pow(cos(interpangle - angle), 4.0f), 0.0f);

			tcomplex* d_Nprojft = d_projft + index * elementsproj + ylocal * (dims.x / 2 + 1);

			cuComplex value = make_cuComplex(0.0f, 0.0f);
			float xfrac = dx - floor(dx);
			int x1 = (int)floor(dx);

			float fac = xfrac * ((2.0f - xfrac) * xfrac - 1.0f);
			value.x += fac * d_Nprojft[max(0, x1 - 1)].x; 
			value.y += fac * d_Nprojft[max(0, x1 - 1)].y; 

			fac = xfrac * xfrac * (3.0f * xfrac - 5.0f) + 2.0f;
			value.x += fac * d_Nprojft[x1].x; 
			value.y += fac * d_Nprojft[x1].y; 

			fac = xfrac * ((4.0f - 3.0f * xfrac) * xfrac + 1.0f);
			value.x += fac * d_Nprojft[min(dims.x / 2, x1 + 1)].x; 
			value.y += fac * d_Nprojft[min(dims.x / 2, x1 + 1)].y;

			fac = (xfrac - 1.0f) * xfrac * xfrac;
			value.x += fac * d_Nprojft[min(dims.x / 2, x1 + 2)].x; 
			value.y += fac * d_Nprojft[min(dims.x / 2, x1 + 2)].y;

			value.x *= 0.5f;
			value.y *= isconj ? -0.5f : 0.5f;

			sumre += value.x * d;
			sumim += value.y * d;
			samples += d;
		}
		if(samples > 0.0f)
		{
			d_interpolated[id].x = samples;// (tfloat)sumre / samples;
			d_interpolated[id].y = (tfloat)sumim / samples;
		}
		else
		{
			d_interpolated[id].x = 0.0f;
			d_interpolated[id].y = 0.0f;
		}
		//d_interpolated[id].x = samples;
	}
}