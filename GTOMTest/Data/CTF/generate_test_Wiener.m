%Case 1:
ctf = tom_ctf1d(129, 1e-10, 300e3, 2e-3, 2.2e-3, -1e-6, 0.07, 0, 0, 0);
fid = fopen('Input1_Wiener.bin','W');
fwrite(fid,ctf,'single');
fclose(fid);

fsc = zeros(129,1);
fid = fopen('Input2_Wiener.bin','W');
fwrite(fid,fsc,'single');
fclose(fid);