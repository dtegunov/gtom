#include "../Prerequisites.cuh"
#include "../Functions.cuh"


////////////////////////////
//CUDA kernel declarations//
////////////////////////////




///////////////////////////////////////
//Equivalent of TOM's tom_peak method//
///////////////////////////////////////

void d_Peak(tfloat* d_input, tfloat3* d_positions, tfloat* d_values, int3 dims, T_PEAK_MODE mode, cufftHandle* planforw, cufftHandle* planback, int batch)
{
	tuple2<tfloat, size_t>* d_integerindices;
	cudaMalloc((void**)&d_integerindices, batch * sizeof(tuple2<tfloat, size_t>));

	if(batch <= 1)
		d_Max(d_input, d_integerindices, Elements(dims), batch);
	else
		d_MaxMonolithic(d_input, d_integerindices, Elements(dims), batch);
	tuple2<tfloat, size_t>* h_integerindices = (tuple2<tfloat, size_t>*)MallocFromDeviceArray(d_integerindices, batch * sizeof(tuple2<tfloat, size_t>));

	tfloat3* h_positions = (tfloat3*)malloc(batch * sizeof(tfloat3));
	tfloat* h_values = (tfloat*)malloc(batch * sizeof(tfloat));

	for (int b = 0; b < batch; b++)
	{
		size_t index = h_integerindices[b].t2;
		size_t z = index / (dims.x * dims.y);
		index -= z * (dims.x * dims.y);
		size_t y = index / dims.x;
		index -= y * dims.x;
			
		h_positions[b] = tfloat3((tfloat)index, (tfloat)y, (tfloat)z);
		h_values[b] = h_integerindices[b].t1;
	}
	if(mode == T_PEAK_MODE::T_PEAK_INTEGER)
	{
		cudaMemcpy(d_positions, h_positions, batch * sizeof(tfloat3), cudaMemcpyHostToDevice);
		cudaMemcpy(d_values, h_values, batch * sizeof(tfloat), cudaMemcpyHostToDevice);
	}
	else if(mode == T_PEAK_MODE::T_PEAK_SUBCOARSE)
	{
		int samples = 9;
		for (int i = 0; i < DimensionCount(dims); i++)	//Samples shouldn't be bigger than smallest relevant dimension
			samples = min(samples, ((int*)&dims)[i]);
		int subdivisions = 105;							//Theoretical precision is 1/subdivisions, 105 = 3*5*7 -> good for FFT
		int centerindex = samples / 2 * subdivisions;

		tfloat* d_original;
		cudaMalloc((void**)&d_original, samples * sizeof(tfloat));
		tfloat* d_interpolated;
		cudaMalloc((void**)&d_interpolated, samples * subdivisions * sizeof(tfloat));
		tuple2<tfloat, size_t>* d_maxtuple;
		cudaMalloc((void**)&d_maxtuple, sizeof(tuple2<tfloat, size_t>));
		tuple2<tfloat, size_t>* h_maxtuple = (tuple2<tfloat, size_t>*)malloc(sizeof(tuple2<tfloat, size_t>));

		for (int b = 0; b < batch; b++)
		{
			int3 coarseposition = toInt3((int)h_positions[b].x, (int)h_positions[b].y, (int)h_positions[b].z);

			//Interpolate along 1st dimension
			d_Extract(d_input + Elements(dims) * b, d_original, dims, toInt3(samples, 1, 1), coarseposition);
			d_Scale(d_original, d_interpolated, toInt3(samples, 1, 1), toInt3(samples * subdivisions, 1 , 1), T_INTERP_MODE::T_INTERP_FOURIER, planforw, planback);
			d_Max(d_interpolated, d_maxtuple, samples * subdivisions);
			cudaMemcpy(h_maxtuple, d_maxtuple, sizeof(tuple2<tfloat, size_t>), cudaMemcpyDeviceToHost);
			h_values[b] = max(h_values[b], (*h_maxtuple).t1);
			h_positions[b].x += (tfloat)((int)(*h_maxtuple).t2 - centerindex) / (tfloat)subdivisions;

			//Interpolate along 2st dimension
			if(dims.y > 1)
			{
				d_Extract(d_input + Elements(dims) * b, d_original, dims, toInt3(1, samples, 1), coarseposition);
				d_Scale(d_original, d_interpolated, toInt3(samples, 1, 1), toInt3(samples * subdivisions, 1 , 1), T_INTERP_MODE::T_INTERP_FOURIER, planforw, planback);
				d_Max(d_interpolated, d_maxtuple, samples * subdivisions);
				cudaMemcpy(h_maxtuple, d_maxtuple, sizeof(tuple2<tfloat, size_t>), cudaMemcpyDeviceToHost);
				h_values[b] = max(h_values[b], (*h_maxtuple).t1);
				h_positions[b].y += (tfloat)((int)(*h_maxtuple).t2 - centerindex) / (tfloat)subdivisions;
			}

			//Interpolate along 3rd dimension
			if(dims.z > 1)
			{
				d_Extract(d_input + Elements(dims) * b, d_original, dims, toInt3(1, 1, samples), coarseposition);
				d_Scale(d_original, d_interpolated, toInt3(samples, 1, 1), toInt3(samples * subdivisions, 1 , 1), T_INTERP_MODE::T_INTERP_FOURIER, planforw, planback);
				d_Max(d_interpolated, d_maxtuple, samples * subdivisions);
				cudaMemcpy(h_maxtuple, d_maxtuple, sizeof(tuple2<tfloat, size_t>), cudaMemcpyDeviceToHost);
				h_values[b] = max(h_values[b], (*h_maxtuple).t1);
				h_positions[b].z += (tfloat)((int)(*h_maxtuple).t2 - centerindex) / (tfloat)subdivisions;
			}
		}

		cudaFree(d_original);
		cudaFree(d_interpolated);
		cudaFree(d_maxtuple);
		free(h_maxtuple);

		cudaMemcpy(d_positions, h_positions, batch * sizeof(tfloat3), cudaMemcpyHostToDevice);
		cudaMemcpy(d_values, h_values, batch * sizeof(tfloat), cudaMemcpyHostToDevice);
	}
	else if(mode == T_PEAK_MODE::T_PEAK_SUBFINE)
	{
		int samples = DimensionCount(dims) < 3 ? 9 : 5;	//Region around the peak to be extracted
		for (int i = 0; i < DimensionCount(dims); i++)	//Samples shouldn't be bigger than smallest relevant dimension
			samples = min(samples, ((int*)&dims)[i]);
		int subdivisions = DimensionCount(dims) < 3 ? 105 : 63;		//Theoretical precision is 1/subdivisions; scaling 3D map is more expensive, thus less precision there
		int centerindex = samples / 2 * subdivisions;	//Where the peak is within the extracted, up-scaled region

		tfloat* d_original;
		cudaMalloc((void**)&d_original, pow(samples, DimensionCount(dims)) * sizeof(tfloat));
		tfloat* d_interpolated;
		cudaMalloc((void**)&d_interpolated, pow(samples * subdivisions, DimensionCount(dims)) * sizeof(tfloat));
		tuple2<tfloat, size_t>* d_maxtuple;
		cudaMalloc((void**)&d_maxtuple, sizeof(tuple2<tfloat, size_t>));
		tuple2<tfloat, size_t>* h_maxtuple = (tuple2<tfloat, size_t>*)malloc(sizeof(tuple2<tfloat, size_t>));

		int indexdifference;
		for (int b = 0; b < batch; b++)
		{
			int3 coarseposition = toInt3((int)h_positions[b].x, (int)h_positions[b].y, (int)h_positions[b].z);

			d_Extract(d_input + Elements(dims) * b, d_original, dims, toInt3(samples, min(dims.y, samples), min(dims.z, samples)), coarseposition);
			d_Scale(d_original, 
					d_interpolated, 
					toInt3(samples, min(dims.y, samples), min(dims.z, samples)), 
					toInt3(samples * subdivisions, dims.y == 1 ? 1 : samples * subdivisions, dims.z == 1 ? 1 : samples * subdivisions), 
					T_INTERP_MODE::T_INTERP_FOURIER,
					planforw,
					planback);
			d_Max(d_interpolated, d_maxtuple, pow(samples * subdivisions, DimensionCount(dims)));
			cudaMemcpy(h_maxtuple, d_maxtuple, sizeof(tuple2<tfloat, size_t>), cudaMemcpyDeviceToHost);

			h_values[b] = max(h_values[b], (*h_maxtuple).t1);

			size_t index = (*h_maxtuple).t2;
			size_t z = index / (samples * samples * subdivisions * subdivisions);
			index -= z * (samples * samples * subdivisions * subdivisions);
			size_t y = index / (samples * subdivisions);
			index -= y * (samples * subdivisions);

			h_positions[b].x += (tfloat)((int)index - centerindex) / (tfloat)subdivisions;
			if(dims.y > 1)
				h_positions[b].y += (tfloat)((int)y - centerindex) / (tfloat)subdivisions;
			if(dims.z > 1)
				h_positions[b].z += (tfloat)((int)z - centerindex) / (tfloat)subdivisions;
		}

		cudaFree(d_original);
		cudaFree(d_interpolated);
		cudaFree(d_maxtuple);
		free(h_maxtuple);

		cudaMemcpy(d_positions, h_positions, batch * sizeof(tfloat3), cudaMemcpyHostToDevice);
		cudaMemcpy(d_values, h_values, batch * sizeof(tfloat), cudaMemcpyHostToDevice);
	}


	free(h_integerindices);
	free(h_positions);
	free(h_values);
	cudaFree(d_integerindices);
}

void d_PeakMakePlans(int3 dims, T_PEAK_MODE mode, cufftHandle* planforw, cufftHandle* planback)
{
	if (mode == T_PEAK_SUBFINE)
	{
		int samples = DimensionCount(dims) < 3 ? 9 : 5;
		for (int i = 0; i < DimensionCount(dims); i++)	//Samples shouldn't be bigger than smallest relevant dimension
			samples = min(samples, ((int*)&dims)[i]);
		int subdivisions = DimensionCount(dims) < 3 ? 105 : 63;		//Theoretical precision is 1/subdivisions

		int3 dimsold = toInt3(samples, min(dims.y, samples), min(dims.z, samples));
		int3 dimsnew = toInt3(samples * subdivisions, dims.y == 1 ? 1 : samples * subdivisions, dims.z == 1 ? 1 : samples * subdivisions);

		*planforw = d_FFTR2CGetPlan(DimensionCount(dims), dimsold);
		*planback = d_IFFTC2CGetPlan(DimensionCount(dims), dimsnew);
	}
	else if (mode == T_PEAK_SUBCOARSE)
	{
		int samples = 9;
		for (int i = 0; i < DimensionCount(dims); i++)	//Samples shouldn't be bigger than smallest relevant dimension
			samples = min(samples, ((int*)&dims)[i]);
		int subdivisions = 105;							//Theoretical precision is 1/subdivisions, 105 = 3*5*7 -> good for FFT

		int3 dimsold = toInt3(samples, 1, 1);
		int3 dimsnew = toInt3(samples * subdivisions, 1, 1);

		*planforw = d_FFTR2CGetPlan(DimensionCount(dims), dimsold);
		*planback = d_IFFTC2CGetPlan(DimensionCount(dims), dimsnew);
	}
}

////////////////
//CUDA kernels//
////////////////

