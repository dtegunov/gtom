function [simmatrix, samplesmatrix, rotationsmatrix, translationsmatrix] = gtom_similarity_matrix_3d( volumes, psf, angularspacing, maxtheta )

[simmatrix, samplesmatrix, rotationsmatrix, translationsmatrix] = cgtom_similarity_matrix_3d(volumes, psf, angularspacing, maxtheta);

end