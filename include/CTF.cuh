#include "Prerequisites.cuh"
#include "Angles.cuh"
#include "Optimization.cuh"

#ifndef CTF_CUH
#define CTF_CUH

namespace gtom
{
	// All lengths in meters
	struct CTFParams
	{
		tfloat pixelsize;
        tfloat pixeldelta;
        tfloat pixelangle;
		tfloat Cs;
		tfloat voltage;
		tfloat defocus;
		tfloat astigmatismangle;
		tfloat defocusdelta;
		tfloat amplitude;
		tfloat Bfactor;
		tfloat Bfactordelta;
		tfloat Bfactorangle;
		tfloat scale;
		tfloat phaseshift;

		CTFParams() :
			pixelsize(1e-10),
			pixeldelta(0),
			pixelangle(0),
			Cs(2e-3),
			voltage(300e3),
			defocus(-3e-6),
			astigmatismangle(0),
			defocusdelta(0),
			amplitude(0.07),
			Bfactor(0),
			Bfactordelta(0),
			Bfactorangle(0),
			scale(1.0),
			phaseshift(0) {}
	};

	struct CTFFitParams
	{
		tfloat3 pixelsize;
		tfloat3 pixeldelta;
		tfloat3 pixelangle;
		tfloat3 Cs;
		tfloat3 voltage;
		tfloat3 defocus;
		tfloat3 astigmatismangle;
		tfloat3 defocusdelta;
		tfloat3 amplitude;
		tfloat3 Bfactor;
		tfloat3 scale;
		tfloat3 phaseshift;

		int2 dimsperiodogram;
		int maskinnerradius;
		int maskouterradius;

		CTFFitParams() :
			pixelsize(0),
			pixeldelta(0),
			pixelangle(0),
			Cs(0),
			voltage(0),
			defocus(0),
			astigmatismangle(0),
			defocusdelta(0),
			amplitude(0),
			Bfactor(0),
			scale(0),
			phaseshift(0),
			dimsperiodogram(toInt2(512, 512)),
			maskinnerradius(1),
			maskouterradius(128) {}

		CTFFitParams(CTFParams p) :
			pixelsize(p.pixelsize),
			pixeldelta(p.pixeldelta),
			pixelangle(p.pixelangle),
			Cs(p.Cs),
			voltage(p.voltage),
			defocus(p.defocus),
			astigmatismangle(p.astigmatismangle),
			defocusdelta(p.defocusdelta),
			amplitude(p.amplitude),
			Bfactor(p.Bfactor),
			scale(p.scale),
			phaseshift(p.phaseshift),
			dimsperiodogram(toInt2(512, 512)),
			maskinnerradius(1),
			maskouterradius(128) {}
	};

	// All lengths in Angstrom
	struct CTFParamsLean
	{
		tfloat ny;
		tfloat pixelsize;
		tfloat pixeldelta;
		tfloat pixelangle;
		tfloat lambda;
		tfloat defocus;
		tfloat astigmatismangle;
		tfloat defocusdelta;
		tfloat Cs;
		tfloat scale;
		tfloat phaseshift;
		tfloat K1, K2, K3;
		tfloat Bfactor, Bfactordelta, Bfactorangle;

		CTFParamsLean(CTFParams p, int3 dims) :
			ny(1.0f / (dims.z > 1 ? (tfloat)dims.x * (p.pixelsize * 1e10) : (tfloat)dims.x)),
			pixelsize(p.pixelsize * 1e10),
			pixeldelta(p.pixeldelta * 0.5e10),
			pixelangle(p.pixelangle),
			lambda(12.2643247 / sqrt(p.voltage * (1.0 + p.voltage * 0.978466e-6))),
			defocus(p.defocus * 1e10),
			astigmatismangle(p.astigmatismangle),
			defocusdelta(p.defocusdelta * 0.5e10),
			Cs(p.Cs * 1e10),
			scale(p.scale),
			phaseshift(p.phaseshift),
			K1(PI * lambda),
			K2(PIHALF * (p.Cs * 1e10) * lambda * lambda * lambda),
			K3(atan(p.amplitude / sqrt(1 - p.amplitude * p.amplitude))),
			Bfactor(p.Bfactor * 0.25e20),
			Bfactordelta(p.Bfactordelta * 0.25e20),
			Bfactorangle(p.Bfactorangle) {}
	};

	class CTFTiltParams
	{
	public:
		tfloat imageangle;
		tfloat2 stageangle;
		tfloat2 specimenangle;
		CTFParams centerparams;

		CTFTiltParams(tfloat3 _angles, CTFParams _centerparams)
		{
			imageangle = _angles.z;
			stageangle = tfloat2(_angles.x, _angles.y);
			specimenangle = tfloat2(0);
			centerparams = _centerparams;
		}

		CTFTiltParams(tfloat _imageangle, tfloat2 _stageangle, tfloat2 _specimenangle, CTFParams _centerparams)
		{
			imageangle = _imageangle;
			stageangle = _stageangle;
			specimenangle = _specimenangle;
			centerparams = _centerparams;
		}

		void GetZGrid2D(int2 dimsimage, int2 dimsregion, int3* h_positions, uint npoints, float* h_zvalues)
		{
			float2 imagecenter = make_float2(dimsimage.x / 2.0f, dimsimage.y / 2.0f);
			float2 regioncenter = make_float2(dimsregion.x / 2.0f, dimsregion.y / 2.0f);

			glm::mat2 transform2d = Matrix2Rotation(imageangle);
			glm::mat3 transform3d = Matrix3RotationInPlaneAxis(stageangle.x, stageangle.y) * Matrix3RotationInPlaneAxis(specimenangle.x, specimenangle.y);
			glm::vec3 planenormal = transform3d * glm::vec3(0, 0, 1);

			for (uint n = 0; n < npoints; n++)
			{
				glm::vec2 flatcoords = glm::vec2(h_positions[n].x + regioncenter.x - imagecenter.x,
					h_positions[n].y + regioncenter.y - imagecenter.y) * centerparams.pixelsize;
				flatcoords = transform2d * flatcoords;
				float d = glm::dot(glm::vec3(-flatcoords.x, -flatcoords.y, 0), planenormal) / planenormal.z;	// Distance from plane along a vertical line (0, 0, 1)
				h_zvalues[n] = centerparams.defocus + d;
			}
		}

		void GetParamsGrid2D(int2 dimsimage, int2 dimsregion, int3* h_positions, uint npoints, CTFParams* h_params)
		{
			float* h_zvalues = (float*)malloc(npoints * sizeof(float));
			this->GetZGrid2D(dimsimage, dimsregion, h_positions, npoints, h_zvalues);

			for (uint n = 0; n < npoints; n++)
			{
				CTFParams pointparams = centerparams;
				pointparams.defocus = h_zvalues[n];
				h_params[n] = pointparams;
			}

			free(h_zvalues);
		}
	};

	template<bool ampsquared, bool ignorefirstpeak> __device__ tfloat d_GetCTF(tfloat r, tfloat angle, tfloat gammacorrection, CTFParamsLean p)
	{
		tfloat r2 = r * r;
		tfloat r4 = r2 * r2;
				
		tfloat deltaf = p.defocus + p.defocusdelta * __cosf((tfloat)2 * (angle - p.astigmatismangle));
		tfloat gamma = p.K1 * deltaf * r2 + p.K2 * r4 - p.phaseshift - p.K3 + gammacorrection;
		tfloat retval;
		if (ignorefirstpeak && abs(gamma) < PI / 2)
			retval = 1;
		else
			retval = -__sinf(gamma);

		if (p.Bfactor != 0 || p.Bfactordelta != 0)
		{
			tfloat Bfacaniso = p.Bfactor;
			if (p.Bfactordelta != 0)
				Bfacaniso += p.Bfactordelta * __cosf((tfloat)2 * (angle - p.Bfactorangle));

			retval *= __expf(Bfacaniso * r2);
		}

		if (ampsquared)
			retval = abs(retval);

		retval *= p.scale;

		return retval;
	}

	template<bool dummy> __device__ float2 d_GetCTFComplex(tfloat r, tfloat angle, tfloat gammacorrection, CTFParamsLean p, bool reverse)
	{
		tfloat r2 = r * r;
		tfloat r4 = r2 * r2;

		tfloat deltaf = p.defocus + p.defocusdelta * cos((tfloat)2 * (angle - p.astigmatismangle));
		tfloat gamma = p.K1 * deltaf * r2 + p.K2 * r4 - p.phaseshift - p.K3 + PI / 2 + gammacorrection;
		float2 retval = make_float2(cos(gamma), sin(gamma));
		if (reverse)
			retval.y *= -1;

		if (p.Bfactor != 0 || p.Bfactordelta != 0)
		{
			tfloat Bfacaniso = p.Bfactor;
			if (p.Bfactordelta != 0)
				Bfacaniso += p.Bfactordelta * cos((tfloat)2 * (angle - p.Bfactorangle));

			retval *= exp(Bfacaniso * r2);
		}

		retval *= p.scale;

		return retval;
	}

	//AliasingCutoff.cu:
	uint CTFGetAliasingCutoff(CTFParams params, uint sidelength);

	//CommonPSF.cu:
	void d_ForceCommonPSF(tcomplex* d_inft1, tcomplex* d_inft2, tcomplex* d_outft1, tcomplex* d_outft2, tfloat* d_psf1, tfloat* d_psf2, tfloat* d_commonpsf, uint n, bool same2, int batch);

	//Correct.cu:
	void d_CTFCorrect(tcomplex* d_input, int3 dimsinput, CTFParams params, tcomplex* d_output);

	//Decay.cu:
	void d_CTFDecay(tfloat* d_input, tfloat* d_output, int2 dims, int degree, int stripwidth);

	//Fit.cu:
	void d_CTFFitCreateTarget2D(tfloat* d_image, int2 dimsimage, CTFParams params, CTFFitParams fp, float overlapfraction, tfloat* d_ps2dpolar, float2* d_ps2dcoords);
	void d_CTFFitCreateTarget2D(tfloat* d_image, int2 dimsimage, int3* d_origins, CTFParams* h_params, int norigins, CTFFitParams fp, tfloat* d_ps2dpolar, float2* d_ps2dcoords, bool sumtoone = false, tfloat* d_outps1dmin = NULL, tfloat* d_outps1dmax = NULL);
	void d_CTFFitCreateTarget1D(tfloat* d_ps2dpolar, float2* d_ps2dcoords, int2 dimspolar, CTFParams* h_params, int norigins, CTFFitParams fp, tfloat* d_ps1d, float2* d_ps1dcoords);
	void d_CTFFit(tfloat* d_target, float2* d_targetcoords, int2 dimstarget, CTFParams* h_startparams, uint ntargets, CTFFitParams p, int refinements, std::vector<std::pair<tfloat, CTFParams> > &fits, tfloat &score, tfloat &mean, tfloat &stddev);
	void d_CTFFit(tfloat* d_image, int2 dimsimage, float overlapfraction, CTFParams startparams, CTFFitParams fp, int refinements, CTFParams &fit, tfloat &score, tfloat &mean, tfloat &stddev);
	void AddCTFParamsRange(std::vector<std::pair<tfloat, CTFParams> > &v_params, CTFFitParams p);
	void h_CTFFitEnvelope(tfloat* h_input, uint diminput, tfloat* h_envelopemin, tfloat* h_envelopemax, char peakextent, uint outputstart, uint outputend, uint batch);

	//InterpolateIrregular.cu:
	void Interpolate1DOntoGrid(std::vector<tfloat2> sortedpoints, tfloat* h_output, uint gridstart, uint gridend);

	//Periodogram.cu:
	void d_CTFPeriodogram(tfloat* d_image, int2 dimsimage, float overlapfraction, int2 dimsregion, int2 dimspadded, tfloat* d_output2d, bool dopost = true);
	void d_CTFPeriodogram(tfloat* d_image, int2 dimsimage, int3* d_origins, int norigins, int2 dimsregion, int2 dimspadded, tfloat* d_output2d, bool dopost = true, cufftHandle planforw = NULL, tfloat* d_extracted = NULL, tcomplex* d_extractedft = NULL);

	//RotationalAverage.cu:
	void d_CTFRotationalAverage(tfloat* d_re, 
								int2 dimsinput, 
								CTFParams* h_params, 
								tfloat* d_average, 
								ushort freqlow, 
								ushort freqhigh, 
								int batch = 1);
	void d_CTFRotationalAverage(tfloat* d_input, 
								float2* d_inputcoords, 
								uint inputlength, 
								uint sidelength, 
								CTFParams* h_params, 
								tfloat* d_average, 
								ushort freqlow, 
								ushort freqhigh, 
								int batch = 1);
	template<class T> void d_CTFRotationalAverageToTarget(T* d_input, 
														float2* d_inputcoords, 
														uint inputlength, 
														uint sidelength, 
														CTFParams* h_params, 
														CTFParams targetparams, 
														tfloat* d_average, 
														ushort freqlow, 
														ushort freqhigh, 
														int batch = 1);
	void d_CTFRotationalAverageToTargetDeterministic(tfloat* d_input,
													float2* d_inputcoords,
													uint inputlength,
													uint sidelength,
													CTFParams* h_params,
													CTFParams targetparams,
													tfloat* d_average,
													ushort freqlow,
													ushort freqhigh,
													int batch);

	//Simulate.cu:
	void d_CTFSimulate(CTFParams* h_params, float2* d_addresses, float* d_gammacorrection, tfloat* d_output, uint n, bool amplitudesquared = false, bool ignorefirstpeak = false, int batch = 1);
	void d_CTFSimulate(CTFParams* h_params, half2* d_addresses, half* d_output, uint n, bool amplitudesquared = false, bool ignorefirstpeak = false, int batch = 1);
	void d_CTFSimulateComplex(CTFParams* h_params, float2* d_addresses, float* d_gammacorrection, float2* d_output, uint n, bool reverse, int batch = 1);

	//TiltCorrect.cu:
	void d_CTFTiltCorrect(tfloat* d_image, int2 dimsimage, CTFTiltParams tiltparams, tfloat snr, tfloat* d_output);

	//TiltFit.cu:
	void d_AccumulateSpectra(tfloat* d_ps1d, tfloat* d_defoci, uint nspectra, tfloat* d_accumulated, tfloat accumulateddefocus, tfloat* d_perbatchoffsets, CTFParams p, CTFFitParams fp, uint batch = 1);
	void h_CTFTiltFit(tfloat* h_image, int2 dimsimage, uint nimages, float overlapfraction, std::vector<CTFTiltParams> &startparams, CTFFitParams fp, tfloat maxtheta, tfloat2 &specimentilt, tfloat* h_defoci);
	void d_CTFTiltFit(tfloat* d_image, int2 dimsimage, float overlapfraction, CTFTiltParams &startparams, CTFFitParams fp, std::vector<tfloat3> &v_angles, int defocusrefinements, std::vector<tfloat2> &v_results);

	//Wiener.cu:
	void d_CTFWiener(tcomplex* d_input, int3 dimsinput, tfloat* d_fsc, CTFParams* h_params, tcomplex* d_output, tfloat* d_outputweights, uint batch = 1);
	void d_CTFWiener(tcomplex* d_input, int3 dimsinput, tfloat snr, CTFParams* h_params, tcomplex* d_output, tfloat* d_outputweights, uint batch = 1);
}
#endif