#include "Prerequisites.h"

char* LoadArrayFromBinary(string path)
{
	ifstream inputfile(path, ios::in|ios::binary|ios::ate);
	int size = inputfile.tellg();
	char* output = (char*)malloc(size);
	inputfile.seekg(0, ios::beg);
	inputfile.read((char*)output, size);
	inputfile.close();

	return output;
}

double GetMeanAbsoluteError(float* const expected, float* const actual, int n)
{
	double sum = 0.0;
	for(int i = 0; i < n; i++)
		sum += abs((double)expected[i] - (double)actual[i]);
	sum /= (double)n;

	return sum;
}

double GetMeanRelativeError(float* const expected, float* const actual, int n)
{
	double sum = 0.0;
	for(int i = 0; i < n; i++)
	{
		if(expected[i] == 0.0f)
			n--;
		else
			sum += abs((double)expected[i] - (double)actual[i]) / (double)expected[i];
	}

	if(n == 0)
		return -1.0;
	else
		return sum / (double)n;
}