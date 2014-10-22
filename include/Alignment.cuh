#pragma once
#include "Prerequisites.cuh"


/////////////
//Alignment//
/////////////

//Align2D.cu:

/**
* \brief Specifies what kind of alignment should be performed
*/
enum T_ALIGN_MODE
{
	/**Rotational alignment*/
	T_ALIGN_ROT = 1 << 0,
	/**Translational alignment*/
	T_ALIGN_TRANS = 1 << 1,
	/**Rotational and translational alignment*/
	T_ALIGN_BOTH = 3
};

/**
* \brief Aligns a set of images to one or multiple targets and assigns each image to the most likely target
* \param[in] d_input	Images to be aligned
* \param[in] d_targets	Alignment target images
* \param[in] dims	Dimensions of one image (not the stack, i. e. z = 1)
* \param[in] numtargets	Number of alignment targets
* \param[in] h_params	Host array that will, at i * numtargets + t, contain the translation (.x, .y) and rotation (.z) of image i relative to target t
* \param[in] h_membership	Host array that will contain the most likely target ID for each aligned image
* \param[in] h_scores	Host array that will, at i * numtargets + t, contain the cross-correlation value between aligned image i and target t
* \param[in] maxtranslation	Maximum offset allowed for translational alignment
* \param[in] maxrotation	Maximum angular offset allowed for rotational alignment
* \param[in] iterations	Number of iterations if both rotational and translational alignment is performed
* \param[in] mode	Type of 2D alignment to be performed
* \param[in] batch	Number of images to be aligned
*/
void d_Align2D(tfloat* d_input, tfloat* d_targets, int3 dims, int numtargets, tfloat3* h_params, int* h_membership, tfloat* h_scores, int maxtranslation, tfloat maxrotation, int iterations, T_ALIGN_MODE mode, int batch);

//Align3D.cu:

/**
* \brief Aligns a 3D volume to one or multiple target volumes and assigns it to the most likely target
* \param[in] d_input	Volume to be aligned
* \param[in] d_targets	Alignment target volumes
* \param[in] dims	Volume dimensions
* \param[in] numtargets	Number of alignment targets
* \param[in] position	Pointer to variable that will contain the translation for the most likely target
* \param[in] rotation	Pointer to variable that will contain the rotation (Euler angles) for the most likely target
* \param[in] h_membership	Host array that will contain the most likely target ID
* \param[in] h_scores	Host array that will contain the cross-correlation value between aligned volume and each target
* \param[in] h_allpositions	Host array that will contain the translation relative to each alignment target
* \param[in] h_allrotations	Host array that will contain the rotation (Euler angles) relative to each alignment target
* \param[in] maxtranslation	Maximum offset allowed for translational alignment
* \param[in] maxrotation	Maximum angular offset allowed for rotational alignment
* \param[in] rotationstep	Angular sampling step
* \param[in] rotationrefinements	Number of times the alignment should be refined by decreasing rotationstep after the previous iteration
* \param[in] mode	Type of 3D alignment to be performed
*/
void d_Align3D(tfloat* d_input, tfloat* d_targets, int3 dims, int numtargets, tfloat3 &position, tfloat3 &rotation, int* h_membership, tfloat* h_scores, tfloat3* h_allpositions, tfloat3* h_allrotations, int maxtranslation, tfloat3 maxrotation, tfloat rotationstep, int rotationrefinements, T_ALIGN_MODE mode);