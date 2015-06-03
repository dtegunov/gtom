spec1 = tom_ctf1d(512, 1.35e-10, 300e3, 2.0e-3, -1.0e-6, 0.07, 0);
spec2 = tom_ctf1d(512, 1.35e-10, 300e3, 2.0e-3, -3.0e-6, 0.07, 0);

fid = fopen('Input_Accumulate.bin','W');
fwrite(fid,[spec1, spec2],'single');
fclose(fid);