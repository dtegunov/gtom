rng(123);

%Case 1:
dimx=512;
dimy=512;

indata = single(rand(dimx,dimy));

load lena;
indata = tom_rescale(lena,[dimx dimy]);
%indata = tom_rescale(indata,[dimx, dimy, dimz]);

outdata = tom_rescale(lena,[dimx dimy].*2);
for i=1:360
    outdata = imrotate(outdata,1,'bicubic','crop');
end;
outdata = tom_rescale(lena,[dimx dimy]);

fid = fopen('Input_Rotate2D_1.bin','W');
fwrite(fid,indata,'single');
fclose(fid);
fid = fopen('Output_Rotate2D_1.bin','W');
fwrite(fid,outdata,'single');
fclose(fid);