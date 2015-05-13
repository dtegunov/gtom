#include "Prerequisites.h"

TEST(Optimization, OptimizeTomoParams)
{
	cudaDeviceReset();

	//Case 1:
	{
		srand(123);
		string datadir = "D:\\Dev\\RuBisCO\\3NormAndResize\\";
		string paramsdir = "D:\\Dev\\RuBisCO\\4InitParams\\";
		string outputdir = "D:\\Dev\\RuBisCO\\5Align\\";
		string imagename = "L3Tomo3.soft.512.mrc";
		string paramnameroot = "L3Tomo3.512";

		HeaderMRC header = ReadMRCHeader(datadir + imagename);
		int2 dimsimage = toInt2(header.dimensions.x, header.dimensions.y);
		int nimages = header.dimensions.z;
		int3 dimsvolume = toInt3(dimsimage.x, dimsimage.y, dimsimage.x * 10 / 16);
		int maximages = nimages;
		nimages = 20;

		void* h_rawmrc;
		ReadMRC(datadir + imagename, &h_rawmrc);
		tfloat* d_images = MixedToDeviceTfloat(h_rawmrc, header.mode, Elements2(dimsimage) * nimages);
		cudaFreeHost(h_rawmrc);

		tfloat3* h_angles = (tfloat3*)MallocFromBinaryFile(paramsdir + paramnameroot + ".angles");
		tfloat2* h_shifts = (tfloat2*)MallocFromBinaryFile(paramsdir + paramnameroot + ".shifts");
		/*tfloat3* h_angles = (tfloat3*)MallocFromBinaryFile(outputdir + paramnameroot + ".optimangles");
		tfloat2* h_shifts = (tfloat2*)MallocFromBinaryFile(outputdir + paramnameroot + ".optimshifts");*/

		/*tfloat2* h_shifts = (tfloat2*)MallocValueFilled(maximages * 2, (tfloat)0);
		for (int n = 1; n < maximages; n++)
			h_shifts[n] = tfloat2(((float)rand() / (float)RAND_MAX - 0.5f) * 16.0f, ((float)rand() / (float)RAND_MAX - 0.5f) * 16.0f);*/

		tfloat3* h_anglesmin, *h_anglesmax;
		tfloat2* h_shiftsmin, *h_shiftsmax;
		h_anglesmin = (tfloat3*)malloc(maximages * sizeof(tfloat3));
		h_anglesmax = (tfloat3*)malloc(maximages * sizeof(tfloat3));
		h_shiftsmin = (tfloat2*)malloc(maximages * sizeof(tfloat2));
		h_shiftsmax = (tfloat2*)malloc(maximages * sizeof(tfloat2));
		tfloat3 deltaangle = tfloat3(ToRad(1e-5f), ToRad(1e-5f), ToRad(15.0f));
		tfloat deltashift = 32.0f;
		for (int n = 0; n < maximages; n++)
		{
			h_anglesmin[n] = tfloat3(h_angles[n].x - deltaangle.x, h_angles[n].y - deltaangle.y, h_angles[n].z - deltaangle.z);
			h_anglesmax[n] = tfloat3(h_angles[n].x + deltaangle.x, h_angles[n].y + deltaangle.y, h_angles[n].z + deltaangle.z);
			h_shiftsmin[n] = tfloat2(h_shifts[n].x - deltashift, h_shifts[n].y - deltashift);
			h_shiftsmax[n] = tfloat2(h_shifts[n].x + deltashift, h_shifts[n].y + deltashift);
		}
		WriteToBinaryFile(paramsdir + paramnameroot + ".anglesmin", h_anglesmin, maximages * sizeof(tfloat3));
		WriteToBinaryFile(paramsdir + paramnameroot + ".anglesmax", h_anglesmax, maximages * sizeof(tfloat3));
		WriteToBinaryFile(paramsdir + paramnameroot + ".shiftsmin", h_shiftsmin, maximages * sizeof(tfloat2));
		WriteToBinaryFile(paramsdir + paramnameroot + ".shiftsmax", h_shiftsmax, maximages * sizeof(tfloat2));
		/*h_anglesmin = (tfloat3*)MallocFromBinaryFile(paramsdir + paramnameroot + ".anglesmin");
		h_anglesmax = (tfloat3*)MallocFromBinaryFile(paramsdir + paramnameroot + ".anglesmax");
		h_shiftsmin = (tfloat2*)MallocFromBinaryFile(paramsdir + paramnameroot + ".shiftsmin");
		h_shiftsmax = (tfloat2*)MallocFromBinaryFile(paramsdir + paramnameroot + ".shiftsmax");*/

		tfloat finalscore;

		vector<int> indiceshold;
		for (int i = 0; i < 1; i++)
			indiceshold.push_back(i);

		d_OptimizeTomoParamsWBP(d_images, 
								dimsimage, 
								dimsvolume, 
								nimages, 
								indiceshold, 
								h_angles, 
								h_shifts, 
								h_anglesmin, h_anglesmax,
								h_shiftsmin, h_shiftsmax,
								finalscore);

		WriteToBinaryFile(outputdir + paramnameroot + ".optimangles", h_angles, maximages * sizeof(tfloat3));
		WriteToBinaryFile(outputdir + paramnameroot + ".optimshifts", h_shifts, maximages * sizeof(tfloat2));

		free(h_angles);
		free(h_shifts);
		free(h_anglesmin);
		free(h_anglesmax);
		free(h_shiftsmin);
		free(h_shiftsmax);
	}

	cudaDeviceReset();
}