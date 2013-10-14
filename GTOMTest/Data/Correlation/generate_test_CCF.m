%Case 1:
dimx=16;
dimy=16;
indata1 = single(rand(dimx,dimy));
indata2 = tom_shift(indata1, [2 2]);
outdata = tom_corr(indata1, indata2);
fid = fopen('Input1_CCF_1.bin','W');
fwrite(fid,indata1,'single');
fclose(fid);
fid = fopen('Input2_CCF_1.bin','W');
fwrite(fid,indata2,'single');
fclose(fid);
fid = fopen('Output_CCF_1.bin','W');
fwrite(fid,outdata,'single');
fclose(fid);

%Case 2:
dimx=17;
dimy=17;
dimz=17;
indata1 = single(rand(dimx,dimy,dimz));
indata2 = tom_shift(indata1, [3 3 2]);
outdata = tom_corr(indata1, indata2, 'norm');
fid = fopen('Input1_CCF_2.bin','W');
fwrite(fid,indata1,'single');
fclose(fid);
fid = fopen('Input2_CCF_2.bin','W');
fwrite(fid,indata2,'single');
fclose(fid);
fid = fopen('Output_CCF_2.bin','W');
fwrite(fid,outdata,'single');
fclose(fid);