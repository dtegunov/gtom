#include "Prerequisites.cuh"
#include "Angles.cuh"
#include "Optimization.cuh"

#ifndef CTF_CUH
#define CTF_CUH

namespace gtom
{
	struct CTFParams
	{
		tfloat pixelsize;
		tfloat Cs;
		tfloat Cc;
		tfloat voltage;
		tfloat defocus;
		tfloat astigmatismangle;
		tfloat defocusdelta;
		tfloat amplitude;
		tfloat Bfactor;
		tfloat decayCohIll;
		tfloat decayspread;

		CTFParams() :
			pixelsize(1e-10),
			Cs(2e-3),
			Cc(2.2e-3),
			voltage(300e3),
			defocus(-3e-6),
			astigmatismangle(0),
			defocusdelta(0),
			amplitude(0.07),
			Bfactor(0),
			decayCohIll(0.0),
			decayspread(0.8) {}
	};

	struct CTFFitParams
	{
		tfloat3 pixelsize;
		tfloat3 Cs;
		tfloat3 Cc;
		tfloat3 voltage;
		tfloat3 defocus;
		tfloat3 astigmatismangle;
		tfloat3 defocusdelta;
		tfloat3 amplitude;
		tfloat3 Bfactor;
		tfloat3 decayCohIll;
		tfloat3 decayspread;

		int2 dimsperiodogram;
		int maskinnerradius;
		int maskouterradius;

		CTFFitParams() :
			pixelsize(0),
			Cs(0),
			Cc(0),
			voltage(0),
			defocus(0),
			astigmatismangle(0),
			defocusdelta(0),
			amplitude(0),
			Bfactor(0),
			decayCohIll(0),
			decayspread(0),
			dimsperiodogram(toInt2(512, 512)),
			maskinnerradius(1),
			maskouterradius(128) {}

		CTFFitParams(CTFParams p) :
			pixelsize(p.pixelsize),
			Cs(p.Cs),
			Cc(p.Cc),
			voltage(p.voltage),
			defocus(p.defocus),
			astigmatismangle(p.astigmatismangle),
			defocusdelta(p.defocusdelta),
			amplitude(p.amplitude),
			Bfactor(p.Bfactor),
			decayCohIll(p.decayCohIll),
			decayspread(p.decayspread),
			dimsperiodogram(toInt2(512, 512)),
			maskinnerradius(1),
			maskouterradius(128) {}
	};

	struct CTFParamsLean
	{
		double ny;
		double lambda;
		double lambda3;
		double Cs;
		double ccvoltage;
		double defocus;
		double astigmatismangle;
		double defocusdelta;
		double amplitude;
		double Bfactor;
		double decayCohIll;
		double decayspread;
		double q0;

		CTFParamsLean(CTFParams p) :
			ny(0.5f / (p.pixelsize * 1e10)),
			lambda(sqrt(150.4 / (p.voltage * (1.0 + p.voltage / 1022000.0)))),
			lambda3(lambda * lambda * lambda),
			Cs(p.Cs * 1e10),
			ccvoltage(p.Cc * 1e10 / p.voltage),
			defocus(p.defocus * 1e10),
			astigmatismangle(p.astigmatismangle),
			defocusdelta(p.defocusdelta * 0.5e10),
			amplitude(p.amplitude),
			Bfactor(-p.Bfactor * 0.25),
			decayCohIll(p.decayCohIll),
			decayspread(p.decayspread),
			q0(p.decayCohIll / (pow((double)p.Cs * 1e10 * pow((double)lambda, 3.0), 0.25))){}
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

	template<bool ampsquared> __device__ double d_GetCTF(double k, double angle, CTFParamsLean p)
	{
		double k2 = k * k;
		double term1 = p.lambda3 * p.Cs * (k2 * k2);
		if (p.Bfactor == 0.0f)
			p.Bfactor = 1.0f;
		else
			p.Bfactor = exp(p.Bfactor * k2);

		double e_i_k = 1.0f;
		if (p.decayCohIll != 0.0f)
		{
			e_i_k = exp(-(PI * PI) * (p.q0 * p.q0) * pow((double)p.Cs * p.lambda3 * pow(k, 3.0) - p.defocus * p.lambda * k, 2.0));
		}

		double e_e_k = 1.0;
		if (p.decayspread != 0.0)
		{
			double delta_z = p.ccvoltage * p.decayspread;
			e_e_k = exp(-(PI * delta_z * p.lambda * 2.0 * k2));
		}

		double w = PIHALF * (term1 - p.lambda * 2.0 * (p.defocus + p.defocusdelta * sin(2.0 * (angle - p.astigmatismangle))) * k2);
		double amplitude = cos(w);
		double phase = sin(w);

		if (!ampsquared)
			amplitude = e_e_k * e_i_k * p.Bfactor * (sqrt(1 - p.amplitude * p.amplitude) * phase + p.amplitude * amplitude);
		else
		{
			amplitude = (sqrt(1 - p.amplitude * p.amplitude) * phase + p.amplitude * amplitude);
			amplitude = e_e_k * e_i_k * p.Bfactor * amplitude * amplitude;
		}

		return amplitude;
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
	void d_CTFPeriodogram(tfloat* d_image, int2 dimsimage, float overlapfraction, int2 dimsregion, tfloat* d_output2d);
	void d_CTFPeriodogram(tfloat* d_image, int2 dimsimage, int3* d_origins, int norigins, int2 dimsregion, tfloat* d_output2d);

	//RotationalAverage.cu:
	void d_CTFRotationalAverage(tfloat* d_re, int2 dimsinput, CTFParams* h_params, tfloat* d_average, ushort freqlow, ushort freqhigh, int batch = 1);
	void d_CTFRotationalAverage(tfloat* d_input, float2* d_inputcoords, uint inputlength, uint sidelength, CTFParams* h_params, tfloat* d_average, ushort freqlow, ushort freqhigh, int batch = 1);

	//Simulate.cu:
	void d_CTFSimulate(CTFParams* h_params, float2* d_addresses, tfloat* d_output, uint n, bool amplitudesquared = false, int batch = 1);

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