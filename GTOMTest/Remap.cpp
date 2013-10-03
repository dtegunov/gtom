#include "Prerequisites.h"

TEST(Masking, Remap)
{
	cudaDeviceReset();

	int size = 1<<12;
	srand(size);

	int* h_image = (int*)malloc(size * size * sizeof(int));
	for(int i = 0; i < size * size; i++)
		h_image[i] = i;

	/*for(int l = 0; l < size; l++)
	{
		for(int c = 0; c < size; c++)
			printf("%d ", h_image[l * size + c]);
		printf("\n");
	}
	printf("\n");*/

	size_t mapped_desired = 0;
	int* masked_desired = (int*)malloc(size * size * sizeof(int));
	int* h_mask = (int*)malloc(size * size * sizeof(int));
	for(int i = 0; i < size * size; i++)
	{
		h_mask[i] = i % ((rand() % 2) + 1) > 0;
		if(h_mask[i] > 0)
		{
			masked_desired[mapped_desired] = h_image[i];
			mapped_desired++;
		}
	}
		
	/*for(int l = 0; l < size; l++)
	{
		for(int c = 0; c < size; c++)
			printf("%d ", h_mask[l * size + c]);
		printf("\n");
	}
	printf("\n");*/

	intptr_t* map_forward;
	size_t mapped = 0;
	MaskSparseToDense(h_mask, &map_forward, NULL, mapped, size * size);
	ASSERT_EQ(mapped, mapped_desired);

	printf("%d elements mapped\n", mapped);

	int* h_maskedimage = (int*)malloc(mapped * sizeof(int));
	Remap(h_image, map_forward, h_maskedimage, mapped, size * size, 0, 1);

	ASSERT_ARRAY_EQ(h_maskedimage, masked_desired, mapped);

	/*for(int c = 0; c < mapped; c++)
		printf("%d ", h_maskedimage[c]);
	printf("\n");*/

	free(h_image);
	free(map_forward);
	free(h_maskedimage);

	cudaDeviceReset();
}