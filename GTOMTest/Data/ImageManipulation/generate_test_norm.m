%Case 1:
dimx=4;
dimy=3;
dimz=5;
indata = single(rand(dimx,dimy,dimz)).*rand(1);
outdata = tom_norm(indata, 'mean0+1std');
fid = fopen('Input_Norm_1.bin','W');
fwrite(fid,indata,'single');
fclose(fid);
fid = fopen('Output_Norm_1.bin','W');
fwrite(fid,outdata,'single');
fclose(fid);