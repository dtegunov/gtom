rng(123);

%Case 1:
dimx=8;
dimy=8;
dimz=8;

indata = single(rand(dimx,dimy,dimz));
%indata = tom_rescale(indata,[dimx, dimy, dimz]);

outdata = tom_rotate(indata,[0 180 0]);

fid = fopen('Input_Rotate3D_1.bin','W');
fwrite(fid,indata,'single');
fclose(fid);
fid = fopen('Output_Rotate3D_1.bin','W');
fwrite(fid,outdata,'single');
fclose(fid);