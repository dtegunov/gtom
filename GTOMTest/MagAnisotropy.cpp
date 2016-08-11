#include "Prerequisites.h"

TEST(Transformation, MagAnisotropy)
{
	cudaDeviceReset();

	//Case 1:
	{
		int3 dims = toInt3(8, 8, 1);
		tfloat* h_input = (tfloat*)malloc(Elements(dims) * sizeof(tfloat));
		for (uint i = 0; i < Elements(dims); i++)
			h_input[i] = sin((tfloat)(i % 8) / 7.0f * PI);
		tfloat* d_input = (tfloat*)CudaMallocFromHostArray(h_input, Elements(dims) * sizeof(tfloat));
		
		tfloat* d_scaled;
		cudaMalloc((void**)&d_scaled, Elements(dims) * sizeof(tfloat));

		d_MagAnisotropyCorrect(d_input, toInt2(dims.x, dims.y), d_scaled, toInt2(dims.x, dims.y), 1.0f, 1.0f, 0.0f, 8, 1);

		tfloat* h_scaled = (tfloat*)MallocFromDeviceArray(d_scaled, Elements2(dims) * sizeof(tfloat));

		tfloat* h_output = (tfloat*)MallocFromDeviceArray(d_input, Elements(dims) * sizeof(tfloat));
		free(h_output);
		free(h_scaled);
	}


	cudaDeviceReset();
}