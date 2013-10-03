%Case 1:
dimx=256;
indata = single(ones(dimx,1));
outdata = tom_spheremask(indata,126,10);
fid = fopen('Input_Spheremask_1.bin','W');
fwrite(fid,indata,'single');
fclose(fid);
fid = fopen('Output_Spheremask_1.bin','W');
fwrite(fid,outdata,'single');
fclose(fid);

%Case 2:
dimx=256;
dimy=255;
indata = single(ones(dimx,dimy));
outdata = tom_spheremask(indata,126,10);
fid = fopen('Input_Spheremask_2.bin','W');
fwrite(fid,indata,'single');
fclose(fid);
fid = fopen('Output_Spheremask_2.bin','W');
fwrite(fid,outdata,'single');
fclose(fid);

%Case 3:
dimx=256;
dimy=255;
dimz=66;
indata = single(ones(dimx,dimy,dimz));
outdata = tom_spheremask(indata,3,2);
fid = fopen('Input_Spheremask_3.bin','W');
fwrite(fid,indata,'single');
fclose(fid);
fid = fopen('Output_Spheremask_3.bin','W');
fwrite(fid,outdata,'single');
fclose(fid);

%Case 4:
dimx=256;
dimy=256;
indata = single(ones(dimx,dimy));
outdata = tom_spheremask(indata,128,10);
fid = fopen('Input_Spheremask_4.bin','W');
fwrite(fid,indata,'single');
fclose(fid);
fid = fopen('Output_Spheremask_4.bin','W');
fwrite(fid,outdata,'single');
fclose(fid);