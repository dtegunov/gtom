rng(123);

%Case 1:
dimx=64;
dimy=64;

indata = single(rand(dimx,dimy));

load lena;
indata = tom_rescale(lena,[dimx dimy]);
%indata = tom_rescale(indata,[dimx, dimy, dimz]);

outdata = imrotate(indata,30,'bicubic','crop');

fid = fopen('Input_Rotate2D_1.bin','W');
fwrite(fid,indata,'single');
fclose(fid);
fid = fopen('Output_Rotate2D_1.bin','W');
fwrite(fid,outdata,'single');
fclose(fid);