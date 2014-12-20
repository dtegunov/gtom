#pragma once
#include "cufft.h"
#include "Prerequisites.cuh"


//////
//IO//
//////

enum MRC_DATATYPE
{
	MRC_BYTE = 0,
	MRC_SHORT = 1,
	MRC_FLOAT = 2,
	MRC_SHORTCOMPLEX = 3,
	MRC_FLOATCOMPLEX = 4,
	MRC_UNSIGNEDSHORT = 6,
	MRC_RGB = 16
};

struct HeaderMRC
{
	int3 dimensions;
	MRC_DATATYPE mode;
	int3 startsubimage;
	int3 griddimensions;
	float3 pixelsize;
	float3 angles;
	int3 maporder;

	float minvalue;
	float maxvalue;
	float meanvalue;
	int spacegroup;

	int extendedbytes;
	short creatid;

	uchar extradata1[30];

	short nint;
	short nreal;

	uchar extradata2[28];
	
	short idtype;
	short lens;
	short nd1;
	short nd2;
	short vd1;
	short vd2;

	float3 tiltoriginal;
	float3 tiltcurrent;
	float3 origin;

	uchar cmap[4];
	uchar stamp[4];

	float stddevvalue;

	int numlabels;
	uchar labels[10][80];
};

typedef enum EM_DATATYPE: unsigned char
{
	EM_BYTE = 1,
	EM_SHORT = 2,
	EM_SHORTCOMPLEX = 3,
	EM_LONG = 4,
	EM_SINGLE = 5,
	EM_SINGLECOMPLEX = 8,
	EM_DOUBLE = 9,
	EM_DOUBLECOMPLEX = 10
};

struct HeaderEM
{
	uchar machinecoding;
	uchar os9;
	uchar invalid;
	EM_DATATYPE mode;

	int3 dimensions;

	uchar comment[80];
	
	int voltage;
	int Cs;
	int aperture;
	int magnification;
	int ccdmagnification;
	int exposuretime;
	int pixelsize;
	int emcode;
	int ccdpixelsize;
	int ccdwidth;
	int defocus;
	int astigmatism;
	int astigmatismangle;
	int focusincrement;
	int qed;
	int c2intensity;
	int slitwidth;
	int energyoffset;
	int tiltangle;
	int tiltaxis;
	int noname1;
	int noname2;
	int noname3;
	int2 markerposition;
	int resolution;
	int density;
	int contrast;
	int noname4;
	int3 centerofmass;
	int height;
	int noname5;
	int dreistrahlbereich;
	int achromaticring;
	int lambda;
	int deltatheta;
	int noname6;
	int noname7;

	uchar userdata[256];
};

 //mrc.cu:
void ReadMRC(string path, void** data, EM_DATATYPE &datatype, int nframe = 0, bool flipx = false);
void ReadMRCDims(string path, int3 &dims);

//em.cu:
void ReadEM(string path, void** data, EM_DATATYPE &datatype, int nframe = 0);
void ReadEMDims(string path, int3 &dims);

//raw.cu:
void ReadRAW(string path, void** data, EM_DATATYPE datatype, int3 dims, int nframe = 0, size_t headerbytes = 0);
