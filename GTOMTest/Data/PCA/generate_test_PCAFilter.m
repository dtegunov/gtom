%Case 1:
rng(123);
data = rand(10, 6);
[eigenvecs, eigenvals, latent, tsquared, explained, mu] = pca(data, 'Algorithm', 'eig', 'NumComponents', 6, 'Centered', true, 'Economy', false);
eigenvals = eigenvals';
fid = fopen('Input_PCAFilter_eigenvecs.bin','W');
fwrite(fid, eigenvecs, 'single');
fclose(fid);
fid = fopen('Input_PCAFilter_eigenvals.bin','W');
fwrite(fid, eigenvals, 'single');
fclose(fid);
fid = fopen('Input_PCAFilter_data.bin','W');
fwrite(fid, data', 'single');
fclose(fid);

filtdata = tom_pcafilter(data, 4);
fid = fopen('Input_PCAFilter_filtered.bin','W');
fwrite(fid, filtdata', 'single');
fclose(fid);