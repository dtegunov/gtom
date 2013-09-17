#include "Prerequisites.h"

void FFTR2C_Test_Zeros(int const ndimensions, int3 const dimensions)
{
	size_t reallength = dimensions.x * dimensions.y * dimensions.z;
	size_t complexlength = (dimensions.x / 2 + 1) * dimensions.y * dimensions.z;

	tfloat* h_input = MallocZeroFilledFloat(reallength);
	tcomplex* h_output = (tcomplex*)malloc(complexlength * sizeof(tcomplex));

	FFTR2C(h_input, h_output, ndimensions, dimensions);

	ASSERT_ARRAY_EQ((tfloat*)h_output, 0.0f, complexlength * 2);

	free(h_input);
	free(h_output);
}

void IFFTC2R_Test_Zeros(int const ndimensions, int3 const dimensions)
{
	size_t reallength = dimensions.x * dimensions.y * dimensions.z;
	size_t complexlength = (dimensions.x / 2 + 1) * dimensions.y * dimensions.z;

	tcomplex* h_input = (tcomplex*)MallocZeroFilledFloat(complexlength * 2);
	tfloat* h_output = (tfloat*)malloc(reallength * sizeof(tfloat));

	IFFTC2R(h_input, h_output, ndimensions, dimensions);

	ASSERT_ARRAY_EQ((tfloat*)h_output, 0.0f, reallength);

	free(h_input);
	free(h_output);
}

void FFTR2C_Test_Data(string inpath, string outpath, int const ndimensions, int3 const dimensions)
{
	size_t reallength = dimensions.x * dimensions.y * dimensions.z;
	size_t complexlength = (dimensions.x / 2 + 1) * dimensions.y * dimensions.z;

	tfloat* h_input = (tfloat*)MallocFromBinaryFile(inpath);
	tcomplex* h_output = (tcomplex*)malloc(complexlength * sizeof(tcomplex));

	FFTR2C((tfloat*)h_input, (tcomplex*)h_output, ndimensions, dimensions);

	tcomplex* desired_output = (tcomplex*)MallocFromBinaryFile(outpath);
			
	double MeanRelative = GetMeanRelativeError((tfloat*)desired_output, (tfloat*)h_output, complexlength * 2);
	ASSERT_LE(MeanRelative, 1e-5);

	printf("Mean absolute error: %e\n", GetMeanAbsoluteError((tfloat*)desired_output, (tfloat*)h_output, complexlength * 2));
	printf("Mean relative error: %e\n", MeanRelative);

	free(h_input);
	free(h_output);
	free(desired_output);
}

void IFFTC2R_Test_Data(string inpath, string outpath, int const ndimensions, int3 const dimensions)
{
	size_t reallength = dimensions.x * dimensions.y * dimensions.z;
	size_t complexlength = (dimensions.x / 2 + 1) * dimensions.y * dimensions.z;

	tcomplex* h_input = (tcomplex*)MallocFromBinaryFile(inpath);
	tfloat* h_output = (tfloat*)malloc(reallength * sizeof(tfloat));

	IFFTC2R(h_input, h_output, ndimensions, dimensions);

	tfloat* desired_output = (tfloat*)MallocFromBinaryFile(outpath);
				
	double MeanRelative = GetMeanRelativeError((tfloat*)desired_output, (tfloat*)h_output, reallength);
	ASSERT_LE(MeanRelative, 1e-5);
			
	printf("Mean absolute error: %e\n", GetMeanAbsoluteError(desired_output, h_output, reallength));
	printf("Mean relative error: %e\n", MeanRelative);

	free(h_input);
	free(h_output);
	free(desired_output);
}

TEST(FFT, R2C)
{
	//1D, zeros
	for(int i = 512; i <= 16384; i*=2)
	{
		int3 dimensions = {i, 1, 1};
		FFTR2C_Test_Zeros(1, dimensions);

		dimensions.x -= 1;
		FFTR2C_Test_Zeros(1, dimensions);
	}

	//1D, data
	for(int i = 512; i <= 16384; i*=2)
	{
		int3 dimensions = {i, 1, 1};
		stringstream inputname;
		inputname << "Data\\FFT\\Input_1D_" << i << ".bin";
		stringstream outputname;
		outputname << "Data\\FFT\\Output_1D_" << i << ".bin";
		FFTR2C_Test_Data(inputname.str(), outputname.str(), 1, dimensions);
	}

	//2D, zeros	
	for(int i = 512; i <= 16384; i*=2)
	{
		int3 dimensions = {i, i, 1};
		FFTR2C_Test_Zeros(2, dimensions);

		if(i == 16384)	//Without power-of-2 constraint, the plan uses too much memory
			break;

		dimensions.x -= 1;
		dimensions.y -= 1;
		FFTR2C_Test_Zeros(2, dimensions);
	}

	//2D, data
	for(int i = 512; i <= 8192; i*=2)
	{
		int3 dimensions = {i, i, 1};
		stringstream inputname;
		inputname << "Data\\FFT\\Input_2D_" << i << ".bin";
		stringstream outputname;
		outputname << "Data\\FFT\\Output_2D_" << i << ".bin";
		FFTR2C_Test_Data(inputname.str(), outputname.str(), 2, dimensions);
	}

	//3D, zeros	
	for(int i = 64; i <= 512; i*=2)
	{
		int3 dimensions = {i, i, i};
		FFTR2C_Test_Zeros(3, dimensions);

		if(i == 512)	//Without power-of-2 constraint, the plan uses too much memory
			break;

		dimensions.x -= 1;
		dimensions.y -= 1;
		FFTR2C_Test_Zeros(3, dimensions);
	}

	//3D, data
	for(int i = 64; i <= 512; i*=2)
	{
		int3 dimensions = {i, i, i};
		stringstream inputname;
		inputname << "Data\\FFT\\Input_3D_" << i << ".bin";
		stringstream outputname;
		outputname << "Data\\FFT\\Output_3D_" << i << ".bin";
		FFTR2C_Test_Data(inputname.str(), outputname.str(), 3, dimensions);
	}
}

TEST(IFFT, C2R)
{
	//1D, zeros
	for(int i = 512; i <= 16384; i*=2)
	{
		int3 dimensions = {i, 1, 1};
		IFFTC2R_Test_Zeros(1, dimensions);

		dimensions.x -= 1;
		IFFTC2R_Test_Zeros(1, dimensions);
	}

	//1D, data
	for(int i = 512; i <= 16384; i*=2)
	{
		int3 dimensions = {i, 1, 1};
		stringstream inputname;
		inputname << "Data\\FFT\\Output_1D_" << i << ".bin";
		stringstream outputname;
		outputname << "Data\\FFT\\Input_1D_" << i << ".bin";
		IFFTC2R_Test_Data(inputname.str(), outputname.str(), 1, dimensions);
	}

	//2D, zeros	
	for(int i = 512; i <= 16384; i*=2)
	{
		int3 dimensions = {i, i, 1};
		IFFTC2R_Test_Zeros(2, dimensions);

		if(i == 16384)	//Without power-of-2 constraint, the plan uses too much memory
			break;

		dimensions.x -= 1;
		dimensions.y -= 1;
		IFFTC2R_Test_Zeros(2, dimensions);
	}

	//2D, data
	for(int i = 512; i <= 8192; i*=2)
	{
		int3 dimensions = {i, i, 1};
		stringstream inputname;
		inputname << "Data\\FFT\\Output_2D_" << i << ".bin";
		stringstream outputname;
		outputname << "Data\\FFT\\Input_2D_" << i << ".bin";
		IFFTC2R_Test_Data(inputname.str(), outputname.str(), 2, dimensions);
	}

	//3D, zeros	
	for(int i = 64; i <= 512; i*=2)
	{
		int3 dimensions = {i, i, i};
		IFFTC2R_Test_Zeros(3, dimensions);

		if(i == 512)	//Without power-of-2 constraint, the plan uses too much memory
			break;

		dimensions.x -= 1;
		dimensions.y -= 1;
		IFFTC2R_Test_Zeros(3, dimensions);
	}

	//3D, data
	for(int i = 64; i <= 512; i*=2)
	{
		int3 dimensions = {i, i, i};
		stringstream inputname;
		inputname << "Data\\FFT\\Output_3D_" << i << ".bin";
		stringstream outputname;
		outputname << "Data\\FFT\\Input_3D_" << i << ".bin";
		IFFTC2R_Test_Data(inputname.str(), outputname.str(), 3, dimensions);
	}
}