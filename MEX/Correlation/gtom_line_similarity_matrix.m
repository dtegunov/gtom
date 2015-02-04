function simmatrix = gtom_line_similarity_matrix( images, anglesteps, linewidth )

simmatrix = cgtom_line_similarity_matrix(images, anglesteps, linewidth);
simmatrix = max(simmatrix, simmatrix');

end