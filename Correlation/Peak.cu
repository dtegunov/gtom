#include "../Prerequisites.cuh"
#include "../Functions.cuh"


////////////////////////////
//CUDA kernel declarations//
////////////////////////////




///////////////////////////////////////
//Equivalent of TOM's tom_peak method//
///////////////////////////////////////

void d_Peak(tfloat* d_input, tfloat3* d_positions, tfloat* d_values, int3 dims, T_PEAK_MODE mode, int batch)
{
	tuple2<tfloat, size_t>* d_integerindices;
	cudaMalloc((void**)&d_integerindices, batch * sizeof(tuple2<tfloat, size_t>));

	d_Max(d_input, d_integerindices, dims.x * dims.y * dims.z, batch);
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
		
	}


	free(h_integerindices);
	free(h_positions);
	free(h_values);
	cudaFree(d_integerindices);
}


////////////////
//CUDA kernels//
////////////////

