function simmatrix = gtom_similarity_matrix_2d( images, anglesteps )

simmatrix = cgtom_similarity_matrix_2d(images, anglesteps);
simmatrix = max(simmatrix, simmatrix');

end