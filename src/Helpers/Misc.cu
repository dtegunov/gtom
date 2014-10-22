#include "Prerequisites.cuh"

int pow(int base, int exponent)
{
	int result = base;
	for (int i = 0; i < exponent - 1; i++)
		result *= base;
	return result;
}