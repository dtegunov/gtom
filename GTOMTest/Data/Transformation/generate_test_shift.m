%Case 1:
dimx=16;
dimy=16;
indata = single(rand(dimx,dimy));
outdata = tom_shift_fixed(indata,[1 2]);
fid = fopen('Input_Shift_1.bin','W');
fwrite(fid,indata,'single');
fclose(fid);
fid = fopen('Output_Shift_1.bin','W');
fwrite(fid,outdata,'single');
fclose(fid);

%Case 2:
dimx=16;
dimy=16;
indata = single(rand(dimx,dimy));
outdata = tom_shift_fixed(indata,[1.5 2.5]);
fid = fopen('Input_Shift_2.bin','W');
fwrite(fid,indata,'single');
fclose(fid);
fid = fopen('Output_Shift_2.bin','W');
fwrite(fid,outdata,'single');
fclose(fid);