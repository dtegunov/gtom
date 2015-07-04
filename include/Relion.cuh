#include "Prerequisites.cuh"

#ifndef RELION_CUH
#define RELION_CUH

namespace gtom
{
	// DiffRemap.cu:
	void d_rlnDiffRemapDense(tfloat* d_input, tfloat* d_output, uint3* d_orientationindices, uint norientations, uint iclass, uint nparticles, uint nclasses, uint nrot, uint ntrans, uint ntranspadded, tfloat* d_xi2imgs, tfloat* d_sqrtxi2, bool docc);
	void d_rlnDiffRemapSparse(tfloat* d_input, tfloat* d_output, uint3* d_combinations, uint* d_hiddenover, uint elements, uint tileelements, uint weightsperpart, tfloat* d_xi2imgs, tfloat* d_sqrtxi2, bool docc);

	// Project.cu:
	void d_rlnProject(cudaTex t_volumeRe, cudaTex t_volumeIm, int3 dimsvolume, tcomplex* d_proj, int3 dimsproj, uint rmax, glm::mat3* d_matrices, uint batch);

	// SquaredDifferences.cu:
	void d_rlnSquaredDifferences(tcomplex* d_particleft, tfloat* d_minvsigma2, tfloat* d_ctf, int3 dimsparticle, uint nparticles, tcomplex* d_precalcshifts, uint nshifts, tcomplex* d_refft, uint nrefs, uint tile, tfloat* d_diff2, bool dofirstitercc);
	void d_rlnSquaredDifferences180(tcomplex* d_particleft, tfloat* d_minvsigma2, tfloat* d_ctf, int3 dimsparticle, uint nparticles, tcomplex* d_precalcshifts, uint nshifts, tcomplex* d_refft, uint nrefs, uint npsi, uint minref, uint tile, tfloat* d_diff2, bool dofirstitercc);
	void d_rlnSquaredDifferencesSparse(tcomplex* d_particleft, tfloat* d_minvsigma2, tfloat* d_ctf, int3 dimsparticle, tcomplex* d_precalcshifts, tcomplex* d_refft, tfloat* d_diff2, uint3* d_combination, uint ncombinations, uint groupsize, bool dofirstitercc);

	// ConvertWeights.cu:
	void d_rlnConvertWeightsDense(tfloat* d_weights, uint nparticles, uint nclasses, uint nrot, uint ntrans, tfloat* d_pdfrot, tfloat* d_pdftrans, tfloat* d_mindiff2);
	void d_rlnStoreWeightsSort(tfloat* d_input, uint n);
}

#endif
