%Case 1:
dimx=8;
dimy=8;
dimz=8;
indata = tom_spheremask(single(ones(dimx, dimy, dimz)), 3, 0);
rand('seed',123);
indata = single(rand(dimx, dimy, dimz));
%indata(4,4,5) = 10;
%indata(6,6,6) = 20;
outdata = tom_proj3d2c(indata, [0 0]);
fid = fopen('Input_Forward_1.bin','W');
fwrite(fid,indata,'single');
fclose(fid);
fid = fopen('Output_Forward_1.bin','W');
fwrite(fid,outdata,'single');
fclose(fid);