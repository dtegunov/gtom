#include "Prerequisites.cuh"

#ifndef RELION_CUH
#define RELION_CUH

namespace gtom
{
	// Backproject.cu:
	void d_rlnBackproject(tcomplex* d_volumeft, tfloat* d_volumeweights, int3 dimsvolume, tcomplex* d_projft, tfloat* d_projweights, int3 dimsproj, uint rmax, glm::mat3* d_matrices, uint batch);
	void d_rlnBackproject(tcomplex* d_volumeft, tfloat* d_volumeweights, int3 dimsvolume, tcomplex* d_projft, tfloat* d_projweights, int3 dimsproj, uint rmax, tfloat3* h_angles, float supersample, uint batch);

	// ConvertWeights.cu:
	void d_rlnConvertWeightsDense(tfloat* d_weights, uint nparticles, uint nclasses, uint nrot, uint ntrans, tfloat* d_pdfrot, tfloat* d_pdftrans, tfloat* d_mindiff2);
	void d_rlnConvertWeightsSparse(tfloat* d_weightsdense, tfloat* d_weightssparse, uint4* d_combinations, uint nsparse, uint nrot, uint ntrans, tfloat* d_pdfrot, tfloat* d_pdftrans, tfloat* d_mindiff2);
	void d_rlnConvertWeightsSort(tfloat* d_input, uint n);
	void d_rlnConvertWeightsCompact(tfloat* d_input, tfloat* d_output, uint &n);

	// DiffRemap.cu:
	void d_rlnDiffRemapDense(tfloat* d_input, tfloat* d_output, uint3* d_orientationindices, uint norientations, uint iclass, uint nparticles, uint nclasses, uint nrot, uint ntrans, uint ntranspadded, tfloat* d_xi2imgs, tfloat* d_sqrtxi2, bool docc);
	void d_rlnDiffRemapSparse(tfloat* d_input, tfloat* d_output, tfloat* d_mindiff2, uint3* d_combinations, uint* d_hiddenover, uint elements, uint tileelements, uint weightsperpart, uint nparticles, tfloat* d_xi2imgs, tfloat* d_sqrtxi2, bool docc);

	// Misc.cu:
	void d_rlnCreateMinvsigma2s(tfloat* d_output, int* d_mresol, uint elements, tfloat* d_sigma2noise, tfloat sigma2fudge);

	// Project.cu:
	void d_rlnProject(tcomplex* d_volumeft, int3 dimsvolume, tcomplex* d_proj, int3 dimsproj, tfloat3* h_angles, float supersample, uint batch);
	void d_rlnProject(tcomplex* d_volumeft, int3 dimsvolume, tcomplex* d_proj, int3 dimsproj, uint rmax, glm::mat3* d_matrices, uint batch);
	void d_rlnProject(cudaTex t_volumeRe, cudaTex t_volumeIm, int3 dimsvolume, tcomplex* d_proj, int3 dimsproj, tfloat3* h_angles, float supersample, uint batch);
	void d_rlnProject(cudaTex t_volumeRe, cudaTex t_volumeIm, int3 dimsvolume, tcomplex* d_proj, int3 dimsproj, uint rmax, glm::mat3* d_matrices, uint batch);

	// SquaredDifferences.cu:
	void d_rlnSquaredDifferences(tcomplex* d_particleft, tfloat* d_minvsigma2, tfloat* d_ctf, int3 dimsparticle, uint nparticles, tcomplex* d_precalcshifts, uint nshifts, tcomplex* d_refft, uint nrefs, uint tile, tfloat* d_diff2, bool dofirstitercc);
	void d_rlnSquaredDifferences180(tcomplex* d_particleft, tfloat* d_minvsigma2, tfloat* d_ctf, int3 dimsparticle, uint nparticles, tcomplex* d_precalcshifts, uint nshifts, tcomplex* d_refft, uint nrefs, uint npsi, uint minref, uint tile, tfloat* d_diff2, bool dofirstitercc);
	void d_rlnSquaredDifferencesSparse(tcomplex* d_particleft, tfloat* d_minvsigma2, tfloat* d_ctf, int3 dimsparticle, tcomplex* d_precalcshifts, tcomplex* d_refft, tfloat* d_diff2, uint3* d_combination, uint ncombinations, uint groupsize, bool dofirstitercc);

	// StoreWeights.cu:
	void d_rlnStoreWeightsAdd(tcomplex* d_input, tfloat* d_ctf, tcomplex* d_ref, tfloat* d_minvsigma2, 
							  int* d_mresol, 
							  uint elements, uint ntrans, 
							  tfloat* d_weights, tfloat sigweight, tfloat sumweight, 
							  tcomplex* d_output, tfloat* d_outputweights, 
							  tfloat* d_sigma2noise, tfloat* d_normcorrection, tfloat* d_priorclass, 
							  tfloat* d_correctionxa, tfloat* d_correctionaa, bool doscale);
}

#endif
