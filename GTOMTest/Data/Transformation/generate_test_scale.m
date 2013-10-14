%Case 1:
dimx=4;
dimy=4;
newdims=[8 8];
indata = single(rand(dimx,dimy));
outdata = tom_rescale(indata,newdims);
fid = fopen('Input_Scale_1.bin','W');
fwrite(fid,indata,'single');
fclose(fid);
fid = fopen('Output_Scale_1.bin','W');
fwrite(fid,outdata,'single');
fclose(fid);

%Case 2:
dimx=7;
dimy=7;
newdims=[17 17];
indata = single(rand(dimx,dimy));
outdata = tom_rescale(indata,newdims);
fid = fopen('Input_Scale_2.bin','W');
fwrite(fid,indata,'single');
fclose(fid);
fid = fopen('Output_Scale_2.bin','W');
fwrite(fid,outdata,'single');
fclose(fid);

%Case 3:
dimx=16;
dimy=16;
newdims=[8 8];
indata = single(rand(dimx,dimy));
outdata = tom_rescale(indata,newdims);
fid = fopen('Input_Scale_3.bin','W');
fwrite(fid,indata,'single');
fclose(fid);
fid = fopen('Output_Scale_3.bin','W');
fwrite(fid,outdata,'single');
fclose(fid);

%Case 4:
dimx=15;
dimy=15;
newdims=[7 7];
indata = single(rand(dimx,dimy));
outdata = tom_rescale(indata,newdims);
fid = fopen('Input_Scale_4.bin','W');
fwrite(fid,indata,'single');
fclose(fid);
fid = fopen('Output_Scale_4.bin','W');
fwrite(fid,outdata,'single');
fclose(fid);

%Case 5:
dimx=7;
dimy=7;
dimz=7;
newdims=[15 15 15];
indata = single(rand(dimx,dimy,dimz));
outdata = tom_rescale(indata,newdims);
fid = fopen('Input_Scale_5.bin','W');
fwrite(fid,indata,'single');
fclose(fid);
fid = fopen('Output_Scale_5.bin','W');
fwrite(fid,outdata,'single');
fclose(fid);

%Case 6:
dimx=7;
dimy=1;
newdims=[15 1];
indata = single(rand(dimx,dimy));
outdata = tom_rescale(indata,newdims);
fid = fopen('Input_Scale_6.bin','W');
fwrite(fid,indata,'single');
fclose(fid);
fid = fopen('Output_Scale_6.bin','W');
fwrite(fid,outdata,'single');
fclose(fid);