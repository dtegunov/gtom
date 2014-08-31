#include "../Prerequisites.cuh"
#include "../Functions.cuh"


////////////////////////////
//CUDA kernel declarations//
////////////////////////////

__global__ void InterpolateSingleAxisTiltKernel(tcomplex* d_projft, size_t elementsproj, int3 dims, tcomplex* d_interpolated, tfloat* d_weights, tfloat* d_angles, tfloat interpangle, short* d_indices, short npoints, tfloat interpradius, int limity);


////////////////////////////////////////////////////////////////////
//Interpolates a certain image in a tilt series from its neighbors//
////////////////////////////////////////////////////////////////////

void d_InterpolateSingleAxisTilt(tcomplex* d_projft, int3 dimsproj, tcomplex* d_interpolated, tfloat* d_weights, tfloat* h_angles, int interpindex, int maxpoints, tfloat interpradius, int limity)
{
	tfloat interpangle = h_angles[interpindex];		//Angle at which the interpolation should take place
	int npoints = dimsproj.z - 1;//maxpoints;		//Number of samples to use, maximum is number of images in stack - 1
	tfloat* h_factors = (tfloat*)malloc(npoints * sizeof(tfloat));	//Angular offset of the samples
	short* h_indices = (short*)malloc(npoints * sizeof(short));		//Indices of sampling angles in stack

	int n = 0;

	for (int i = 0; i < dimsproj.z; i++)
	{
		if(i == interpindex)
			continue;

		float anglediff = acos(cos(h_angles[i]) * cos(h_angles[interpindex]) + sin(h_angles[i]) * sin(h_angles[interpindex]));		//Dot product between sample slice and interpolant
		float conjangle = ToRad(180.0f) + h_angles[i];
		float conjanglediff = acos(cos(conjangle) * cos(h_angles[interpindex]) + sin(conjangle) * sin(h_angles[interpindex]));		//Dot product between conjugate sample slice and interpolant
		bool isconj = conjanglediff < anglediff;
		float angle = isconj ? (h_angles[i] > 0.0f ? h_angles[i] - ToRad(180.0f) : h_angles[i] + ToRad(180.0f)) : h_angles[i];

		h_factors[n] = angle;
		h_indices[n] = (short)i;
		n++;
	}

	tfloat* d_factors = (tfloat*)CudaMallocFromHostArray(h_factors, n * sizeof(tfloat));
	short* d_indices = (short*)CudaMallocFromHostArray(h_indices, n * sizeof(short));
	free(h_factors);
	free(h_indices);

	int TpB = min(NextMultipleOf((dimsproj.x / 2 + 1) * dimsproj.y, 32), 128);
	dim3 grid = dim3(min(((dimsproj.x / 2 + 1) * dimsproj.y + TpB - 1) / TpB, 8192));
	InterpolateSingleAxisTiltKernel <<<grid, TpB>>> (d_projft, (dimsproj.x / 2 + 1) * dimsproj.y, dimsproj, d_interpolated, d_weights, d_factors, interpangle, d_indices, n, interpradius, limity);

	cudaFree(d_factors);
	cudaFree(d_indices);
}


////////////////
//CUDA kernels//
////////////////

__global__ void InterpolateSingleAxisTiltKernel(tcomplex* d_projft, size_t elementsproj, int3 dims, tcomplex* d_interpolated, tfloat* d_weights, tfloat* d_angles, tfloat interpangle, short* d_indices, short npoints, tfloat interpradius, int limity)
{
	for(size_t id = blockIdx.x * blockDim.x + threadIdx.x; 
		id < elementsproj; 
		id += blockDim.x * gridDim.x)
	{
		float x = (float)(id % (dims.x / 2 + 1));		//id is a 1D index over the entire 2D plane
		float y = (float)(id / (dims.x / 2 + 1));

		float sumre = 0.0f;
		float sumim = 0.0f;
		float samples = 0.0f;

		float coscenter = abs(cos(interpangle));
		float centerx = coscenter * x;
		float centerz = sin(interpangle) * x;

		if (abs(dims.y / 2 - y) > limity)
			for (int n = 0; n < npoints; n++)
			{
				int index = d_indices[n];
				float angle = d_angles[n];
				bool isconj = angle > ToRad(90.0f) || angle < ToRad(-90.0f);	//Outside of half-circle centered around 0
				float anglediff = abs(interpangle - angle);

				if(!(x > 0 && y > 0))		//No need for conjugate on the central line that is the tilt axis
					isconj = false;
				int ylocal = isconj ? dims.y - 1 - (int)y : (int)y;		//Flip y if conjugate

				//Scale the sampling plane along the x axis to accomodate for the fact that the volume is not cubical, but rather thin; 
				//if it were perfectly thin, scaling would be just cos(interpolant)/cos(center) and yield the perfect interpolant sans high frequencies
				float cosangle = abs(cos(angle));
				if (cosangle < 0.001f)
					continue;
				float scalefac = coscenter / cosangle;		
				scalefac = 1.0f;// + (scalefac - 1.0f) * (1.0f - 32.0f / 128.0f);

				float dx = x * scalefac;
			
				//Check if sample is spatially further away from interpolant than allowed by interpradius
				float anglex = dx * cosangle - centerx;
				float anglez = dx * sin(angle) - centerz;
				float d = interpradius - sqrt(anglex * anglex + anglez * anglez);
				if(d <= 0.0f)
					continue;
				d *= max(cos(interpangle - angle), 0.0f);	//Additionally, weight reciprocally to orthogonality

				tcomplex* d_Nprojft = d_projft + index * elementsproj + ylocal * (dims.x / 2 + 1);		//Interpolation will be done only along x axis, so put address to its first element

				//Cubic interpolation using 4 samples along x axis
				cuComplex value = make_cuComplex(0.0f, 0.0f);
				/*float xfrac = dx - floor(dx);
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
				value.y *= isconj ? -0.5f : 0.5f;*/

				value = d_Nprojft[(int)x];
				if (isconj)
					value.y = -value.y;

				sumre += value.x * d;
				sumim += value.y * d;
				samples += d;
			}

		if(samples > 0.0f)
		{
			d_interpolated[id].x = (tfloat)sumre;
			d_interpolated[id].y = (tfloat)sumim;
			d_weights[id] = samples;
		}
		else
		{
			d_interpolated[id].x = 0.0f;
			d_interpolated[id].y = 0.0f;
			d_weights[id] = 0.0f;
		}
	}
}