%Case 1:
dimx=8;
dimy=8;
delta=[1 2];
image1 = single(rand(dimx,dimy));
image2 = tom_shift(image1, delta);
indata = tom_corr(image1, image2);
outdata = delta + [floor(dimx/2)+1 floor(dimy/2)+1] - [1 1];
if(size(outdata,2)==2)
    outdata = [outdata 0];
end;
fid = fopen('Input_Peak_1.bin','W');
fwrite(fid,indata,'single');
fclose(fid);
fid = fopen('Output_Peak_1.bin','W');
fwrite(fid,outdata,'single');
fclose(fid);

%Case 2:
dimx=9;
dimy=9;
delta=[1.78 2.25];
image1 = single(rand(dimx,dimy));
image2 = tom_shift(image1, delta);
indata = tom_corr(image1, image2);
outdata = delta + [floor(dimx/2)+1 floor(dimy/2)+1] - [1 1];
if(size(outdata,2)==2)
    outdata = [outdata 0];
end;
fid = fopen('Input_Peak_2.bin','W');
fwrite(fid,indata,'single');
fclose(fid);
fid = fopen('Output_Peak_2.bin','W');
fwrite(fid,outdata,'single');
fclose(fid);