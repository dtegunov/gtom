%Case 1:
dimx=256;
dimy=1;
indata = single(rand(dimx,dimy));
outdata = tom_rectanglemask(indata,[126 0],0);
fid = fopen('Input_Rectmask_1.bin','W');
fwrite(fid,indata,'single');
fclose(fid);
fid = fopen('Output_Rectmask_1.bin','W');
fwrite(fid,outdata,'single');
fclose(fid);

%Case 2:
dimx=256;
dimy=255;
indata = single(rand(dimx,dimy));
outdata = tom_rectanglemask(indata,[126 126],0);
fid = fopen('Input_Rectmask_2.bin','W');
fwrite(fid,indata,'single');
fclose(fid);
fid = fopen('Output_Rectmask_2.bin','W');
fwrite(fid,outdata,'single');
fclose(fid);

%Case 3:
dimx=256;
dimy=255;
dimz=66;
indata = single(rand(dimx,dimy,dimz));
outdata = tom_rectanglemask(indata,[3 5 6],0);
fid = fopen('Input_Rectmask_3.bin','W');
fwrite(fid,indata,'single');
fclose(fid);
fid = fopen('Output_Rectmask_3.bin','W');
fwrite(fid,outdata,'single');
fclose(fid);

%Case 4:
dimx=256;
dimy=256;
indata = single(rand(dimx,dimy));
outdata = tom_rectanglemask(indata,[17 16],10);
fid = fopen('Input_Rectmask_4.bin','W');
fwrite(fid,indata,'single');
fclose(fid);
fid = fopen('Output_Rectmask_4.bin','W');
fwrite(fid,outdata,'single');
fclose(fid);