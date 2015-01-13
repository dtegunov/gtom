%Case 1:
data = reshape(1:60, 10, 6);
data = rand(10, 6);
[eigenval, eigenvec, cent] = pca(data, 4);
eigenvec = eigenvec';
eigenval = eigenval';
fid = fopen('Input_PCA.bin','W');
fwrite(fid,data','single');
fclose(fid);
